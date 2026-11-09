"""Pilot-aware joint channel estimation and detection for 2x2 PDSCH."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _as_complex(value: Tensor) -> Tensor:
    if value.shape[-1] != 2:
        raise ValueError("Complex tensors must use a final [real, imag] dimension.")
    return torch.view_as_complex(value.contiguous())


def _as_real(value: Tensor) -> Tensor:
    return torch.view_as_real(value)


def lmmse_equalize(
    rx_grid: Tensor,
    channel_est: Tensor,
    noise_var: Tensor,
    valid_re_mask: Tensor | None = None,
) -> Tensor:
    """Diagnostic LMMSE baseline; never used by the joint neural model."""

    y = _as_complex(rx_grid.float())
    h = _as_complex(channel_est.float())
    batch, symbols, subcarriers, num_rx, num_layers = h.shape
    if y.shape != (batch, symbols, subcarriers, num_rx):
        raise ValueError("rx_grid and channel_est dimensions are inconsistent.")
    h_flat = h.reshape(-1, num_rx, num_layers)
    y_flat = y.reshape(-1, num_rx, 1)
    n0 = noise_var.clamp_min(1e-8)[:, None, None].expand(
        batch, symbols, subcarriers
    ).reshape(-1)
    gram = h_flat.mH @ h_flat
    eye = torch.eye(num_layers, dtype=gram.dtype, device=gram.device)
    estimate = torch.linalg.solve(
        gram + n0[:, None, None].to(gram.dtype) * eye,
        h_flat.mH @ y_flat,
    ).squeeze(-1)
    estimate = estimate.reshape(batch, symbols, subcarriers, num_layers)
    if valid_re_mask is not None:
        estimate = estimate * valid_re_mask[..., None].to(estimate.dtype)
    return _as_real(estimate)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(inputs))


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        groups = _group_count(out_channels)
        self.layers = nn.Sequential(
            SeparableConv2d(in_channels, out_channels, dilation),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            SeparableConv2d(out_channels, out_channels, dilation),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class PilotEncoder(nn.Module):
    """Spread sparse DM-RS observations over the full time-frequency grid."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.input = ConvBlock(in_channels, hidden_channels)
        self.dilated = nn.ModuleList(
            ConvBlock(hidden_channels, hidden_channels, dilation)
            for dilation in (1, 2, 4)
        )
        self.output_norm = nn.GroupNorm(
            _group_count(hidden_channels), hidden_channels
        )

    def forward(self, inputs: Tensor) -> Tensor:
        state = self.input(inputs)
        for block in self.dilated:
            state = state + block(state)
        return F.silu(self.output_norm(state))


class ReceiverIteration(nn.Module):
    def __init__(self, hidden_channels: int, evidence_channels: int, dilation: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1), nn.SiLU()
        )
        self.update = ConvBlock(
            hidden_channels * 2 + evidence_channels,
            hidden_channels,
            dilation,
        )
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1), nn.Sigmoid()
        )

    def forward(self, state: Tensor, evidence: Tensor) -> Tensor:
        batch, layers, channels, symbols, subcarriers = state.shape
        message = self.message(
            state.reshape(batch * layers, channels, symbols, subcarriers)
        ).reshape(batch, layers, channels, symbols, subcarriers)
        if layers > 1:
            message = (message.sum(dim=1, keepdim=True) - message) / (layers - 1)
        else:
            message = torch.zeros_like(message)
        update_input = torch.cat((state, message, evidence), dim=2).reshape(
            batch * layers, -1, symbols, subcarriers
        )
        delta = self.update(update_input)
        delta = self.gate(delta) * delta
        return state + delta.reshape(
            batch, layers, channels, symbols, subcarriers
        )


def _nearest_distance(mask: Tensor) -> tuple[Tensor, Tensor]:
    """Separable normalized distance to the nearest pilot in time/frequency."""

    batch, symbols, subcarriers, layers = mask.shape

    def distance_1d(has_pilot: Tensor) -> Tensor:
        length = has_pilot.shape[1]
        index = torch.arange(length, device=mask.device).view(1, length, 1)
        index = index.expand(has_pilot.shape[0], -1, has_pilot.shape[2])
        left = torch.cummax(
            torch.where(has_pilot, index, torch.full_like(index, -length)), dim=1
        ).values
        right = torch.flip(
            torch.cummin(
                torch.flip(
                    torch.where(has_pilot, index, torch.full_like(index, length)),
                    dims=(1,),
                ),
                dim=1,
            ).values,
            dims=(1,),
        )
        distance = torch.minimum(index - left, right - index).clamp(0, length)
        return distance.to(torch.float32) / max(1, length - 1)

    time_distance = distance_1d(mask.any(dim=2))[:, :, None, :].expand(
        batch, symbols, subcarriers, layers
    )
    frequency_distance = distance_1d(mask.any(dim=1))[:, None, :, :].expand(
        batch, symbols, subcarriers, layers
    )
    return time_distance, frequency_distance


class NeuralPDSCHReceiver(nn.Module):
    """Joint receiver using raw RX samples and known DM-RS only."""

    def __init__(
        self,
        num_rx_antennas: int = 2,
        num_layers: int = 2,
        max_bits_per_symbol: int = 10,
        hidden_channels: int = 64,
        condition_channels: int = 16,
        pilot_channels: int = 32,
        num_iterations: int = 6,
        max_num_rbs: int = 275,
        supported_qm: list[int] | tuple[int, ...] = (2, 4, 6, 8, 10),
    ) -> None:
        super().__init__()
        if num_rx_antennas != 2 or num_layers != 2:
            raise ValueError("This implementation is intentionally scoped to 2x2 MIMO.")
        if max(supported_qm) > max_bits_per_symbol:
            raise ValueError("supported_qm exceeds max_bits_per_symbol.")
        self.num_rx_antennas = num_rx_antennas
        self.num_layers = num_layers
        self.max_bits_per_symbol = max_bits_per_symbol
        self.max_num_rbs = max_num_rbs
        self.supported_qm = tuple(int(value) for value in supported_qm)

        self.condition = nn.Sequential(
            nn.Linear(3, condition_channels),
            nn.SiLU(),
            nn.Linear(condition_channels, condition_channels),
            nn.SiLU(),
        )
        raw_evidence_channels = (
            2 * num_rx_antennas
            + 2
            + 1
            + 2 * num_rx_antennas
            + 2
            + 1
            + 1
            + condition_channels
        )
        self.pilot_encoder = PilotEncoder(raw_evidence_channels, pilot_channels)
        evidence_channels = raw_evidence_channels + pilot_channels
        self.state_init = ConvBlock(evidence_channels, hidden_channels)
        dilations = (1, 2, 4)
        self.iterations = nn.ModuleList(
            ReceiverIteration(
                hidden_channels,
                evidence_channels,
                dilations[index % len(dilations)],
            )
            for index in range(num_iterations)
        )
        self.symbol_head = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 3, 1),
        )
        self.bit_heads = nn.ModuleDict(
            {
                str(order): nn.Sequential(
                    ConvBlock(hidden_channels, hidden_channels),
                    nn.Conv2d(hidden_channels, order, 1),
                )
                for order in self.supported_qm
            }
        )
        self.channel_head = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 2 * num_rx_antennas, 1),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NeuralPDSCHReceiver":
        return cls(**config)

    def forward(
        self,
        rx_grid: Tensor,
        dmrs_grid: Tensor,
        dmrs_mask: Tensor,
        noise_var: Tensor,
        data_mask: Tensor,
        qm: Tensor,
        code_rate: Tensor,
        num_rbs: Tensor,
    ) -> dict[str, Tensor]:
        if rx_grid.ndim != 5 or rx_grid.shape[-2:] != (self.num_rx_antennas, 2):
            raise ValueError("rx_grid must have shape [B,S,K,2,2].")
        if data_mask.shape[-1] != self.num_layers:
            raise ValueError("data_mask must have one plane per layer.")
        if dmrs_grid.shape[:-1] != dmrs_mask.shape or dmrs_grid.shape[-1] != 2:
            raise ValueError("dmrs_grid and dmrs_mask dimensions are inconsistent.")
        if data_mask.shape != dmrs_mask.shape:
            raise ValueError("data_mask and dmrs_mask dimensions are inconsistent.")
        if torch.any(data_mask.bool() & dmrs_mask.bool()):
            raise ValueError("data_mask and dmrs_mask must not overlap.")
        if not bool(dmrs_mask.bool().any(dim=(1, 2)).all()):
            raise ValueError("Every sample and spatial layer must contain DM-RS pilots.")
        unsupported = set(qm.detach().cpu().tolist()).difference(self.supported_qm)
        if unsupported:
            raise ValueError(f"Unsupported modulation orders: {sorted(unsupported)}")

        occupied = data_mask.any(dim=-1) | dmrs_mask.any(dim=-1)
        power = rx_grid.square().sum(dim=-1).mean(dim=-1)
        denominator = occupied.sum(dim=(1, 2)).clamp_min(1).to(rx_grid.dtype)
        rms = ((power * occupied).sum(dim=(1, 2)) / denominator).clamp_min(1e-8).sqrt()
        normalized_rx = rx_grid / rms[:, None, None, None, None]
        normalized_n0 = noise_var / rms.square()

        batch, symbols, subcarriers, _, _ = normalized_rx.shape
        rx_features = normalized_rx.permute(0, 3, 4, 1, 2).reshape(
            batch, 2 * self.num_rx_antennas, symbols, subcarriers
        )[:, None].expand(-1, self.num_layers, -1, -1, -1)
        dmrs_features = dmrs_grid.permute(0, 3, 4, 1, 2)
        dmrs_mask_features = dmrs_mask.permute(0, 3, 1, 2)[:, :, None].to(rx_grid.dtype)

        rx_complex = _as_complex(normalized_rx.float())
        dmrs_complex = _as_complex(dmrs_grid.float())
        matched = rx_complex[:, :, :, :, None] * dmrs_complex.conj()[:, :, :, None, :]
        matched = matched * dmrs_mask[:, :, :, None, :].to(matched.dtype)
        matched_features = _as_real(matched).permute(0, 4, 3, 5, 1, 2).reshape(
            batch,
            self.num_layers,
            2 * self.num_rx_antennas,
            symbols,
            subcarriers,
        )
        time_distance, frequency_distance = _nearest_distance(dmrs_mask.bool())
        position_features = torch.stack(
            (time_distance, frequency_distance), dim=-1
        ).permute(0, 3, 4, 1, 2)
        data_features = data_mask.permute(0, 3, 1, 2)[:, :, None].to(rx_grid.dtype)
        log_n0 = normalized_n0.clamp_min(1e-8).log().clamp(-20.0, 10.0)
        log_n0 = log_n0[:, None, None, None, None].expand(
            -1, self.num_layers, -1, symbols, subcarriers
        )
        conditions = torch.stack(
            (
                qm.to(rx_grid.dtype) / self.max_bits_per_symbol,
                code_rate.to(rx_grid.dtype),
                num_rbs.to(rx_grid.dtype) / self.max_num_rbs,
            ),
            dim=-1,
        )
        conditions = self.condition(conditions)[:, None, :, None, None].expand(
            -1, self.num_layers, -1, symbols, subcarriers
        )
        raw_evidence = torch.cat(
            (
                rx_features,
                dmrs_features,
                dmrs_mask_features,
                matched_features,
                position_features,
                data_features,
                log_n0,
                conditions,
            ),
            dim=2,
        )
        flat_raw = raw_evidence.reshape(
            batch * self.num_layers, -1, symbols, subcarriers
        )
        pilot_features = self.pilot_encoder(flat_raw).reshape(
            batch, self.num_layers, -1, symbols, subcarriers
        )
        evidence = torch.cat((raw_evidence, pilot_features), dim=2)
        state = self.state_init(
            evidence.reshape(batch * self.num_layers, -1, symbols, subcarriers)
        ).reshape(batch, self.num_layers, -1, symbols, subcarriers)
        for iteration in self.iterations:
            state = iteration(state, evidence)

        flat_state = state.reshape(batch * self.num_layers, -1, symbols, subcarriers)
        symbol_output = self.symbol_head(flat_state).reshape(
            batch, self.num_layers, 3, symbols, subcarriers
        )
        equalized = symbol_output[:, :, :2].permute(0, 3, 4, 1, 2)
        equalized = equalized * data_mask[..., None].to(equalized.dtype)
        log_symbol_var = symbol_output[:, :, 2].permute(0, 2, 3, 1).clamp(-10.0, 6.0)

        bit_logits = torch.zeros(
            batch,
            symbols,
            subcarriers,
            self.num_layers,
            self.max_bits_per_symbol,
            dtype=state.dtype,
            device=state.device,
        )
        for order in self.supported_qm:
            logits = self.bit_heads[str(order)](flat_state).reshape(
                batch, self.num_layers, order, symbols, subcarriers
            ).permute(0, 3, 4, 1, 2)
            padded = F.pad(logits, (0, self.max_bits_per_symbol - order))
            selected = (qm == order)[:, None, None, None, None].to(padded.dtype)
            bit_logits = bit_logits + padded * selected
        bit_logits = bit_logits * data_mask[..., None].to(bit_logits.dtype)

        channel = self.channel_head(flat_state).reshape(
            batch,
            self.num_layers,
            self.num_rx_antennas,
            2,
            symbols,
            subcarriers,
        ).permute(0, 4, 5, 2, 1, 3)
        channel = channel * rms[:, None, None, None, None, None]
        return {
            "equalized_symbols": equalized,
            "bit_logits": bit_logits,
            "log_symbol_var": log_symbol_var,
            "refined_channel": channel,
        }
