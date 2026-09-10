"""Conditioned iterative convolutional receiver for 2x2 MIMO PDSCH."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


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
    """Return LMMSE symbols as ``[B, S, K, L, 2]``.

    ``rx_grid`` is ``[B,S,K,R,2]`` and ``channel_est`` is
    ``[B,S,K,R,L,2]``. ``noise_var`` is the complex noise power per RE.
    """

    # Complex half-precision linear solves are not portable across CUDA builds.
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
    regularized = gram + n0[:, None, None].to(gram.dtype) * eye
    rhs = h_flat.mH @ y_flat
    estimate = torch.linalg.solve(regularized, rhs).squeeze(-1)
    estimate = estimate.reshape(batch, symbols, subcarriers, num_layers)
    if valid_re_mask is not None:
        estimate = estimate * valid_re_mask[..., None].to(estimate.dtype)
    return _as_real(estimate)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(inputs))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            SeparableConv2d(in_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            SeparableConv2d(out_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class ReceiverIteration(nn.Module):
    def __init__(self, hidden_channels: int, evidence_channels: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1), nn.SiLU()
        )
        self.update = ConvBlock(hidden_channels * 2 + evidence_channels, hidden_channels)
        self.gate = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 1), nn.Sigmoid())

    def forward(self, state: Tensor, evidence: Tensor) -> Tensor:
        # state/evidence: [B,L,C,S,K]. Aggregate all other layers as a graph message.
        batch, layers, channels, symbols, subcarriers = state.shape
        message = self.message(state.reshape(batch * layers, channels, symbols, subcarriers))
        message = message.reshape(batch, layers, channels, symbols, subcarriers)
        if layers > 1:
            message = (message.sum(dim=1, keepdim=True) - message) / (layers - 1)
        else:
            message = torch.zeros_like(message)
        update_input = torch.cat((state, message, evidence), dim=2)
        update_input = update_input.reshape(
            batch * layers, update_input.shape[2], symbols, subcarriers
        )
        delta = self.update(update_input)
        gate = self.gate(delta)
        delta = (gate * delta).reshape(batch, layers, channels, symbols, subcarriers)
        return state + delta


class NeuralPDSCHReceiver(nn.Module):
    """Fully convolutional receiver shared across MCS and RB allocations.

    The physical prior is an LMMSE estimate. A graph-convolutional network then
    exchanges information between layers and predicts a symbol correction plus
    coded-bit logits. Positive bit logits mean bit one.
    """

    def __init__(
        self,
        num_rx_antennas: int = 2,
        num_layers: int = 2,
        max_bits_per_symbol: int = 10,
        hidden_channels: int = 64,
        condition_channels: int = 16,
        num_iterations: int = 6,
        max_num_rbs: int = 275,
    ):
        super().__init__()
        if num_rx_antennas != 2 or num_layers != 2:
            raise ValueError("This implementation is intentionally scoped to 2x2 MIMO.")
        self.num_rx_antennas = num_rx_antennas
        self.num_layers = num_layers
        self.max_bits_per_symbol = max_bits_per_symbol
        self.max_num_rbs = max_num_rbs

        self.condition = nn.Sequential(
            nn.Linear(3, condition_channels),
            nn.SiLU(),
            nn.Linear(condition_channels, condition_channels),
            nn.SiLU(),
        )
        # Per layer: all RX samples, that layer's channel, LMMSE symbol,
        # RE mask, log noise, and the MCS/RB condition.
        evidence_channels = (
            2 * num_rx_antennas
            + 2 * num_rx_antennas
            + 2
            + 1
            + 1
            + condition_channels
        )
        self.state_init = ConvBlock(evidence_channels, hidden_channels)
        self.iterations = nn.ModuleList(
            ReceiverIteration(hidden_channels, evidence_channels)
            for _ in range(num_iterations)
        )
        self.symbol_head = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 3, 1),
        )
        self.bit_head = nn.Sequential(
            ConvBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, max_bits_per_symbol, 1),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NeuralPDSCHReceiver":
        return cls(**config)

    def _normalize(
        self, rx_grid: Tensor, channel_est: Tensor, noise_var: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        re_mask = mask.any(dim=-1).to(rx_grid.dtype)
        power = rx_grid.square().sum(dim=-1).mean(dim=-1)
        denom = re_mask.sum(dim=(1, 2)).clamp_min(1.0)
        rms = ((power * re_mask).sum(dim=(1, 2)) / denom).clamp_min(1e-8).sqrt()
        rx_grid = rx_grid / rms[:, None, None, None, None]
        channel_est = channel_est / rms[:, None, None, None, None, None]
        noise_var = noise_var / rms.square()
        return rx_grid, channel_est, noise_var

    def forward(
        self,
        rx_grid: Tensor,
        channel_est: Tensor,
        noise_var: Tensor,
        data_mask: Tensor,
        qm: Tensor,
        code_rate: Tensor,
        num_rbs: Tensor,
    ) -> dict[str, Tensor]:
        if data_mask.shape[-1] != self.num_layers:
            raise ValueError("data_mask must have one mask plane per layer.")
        rx_grid, channel_est, normalized_n0 = self._normalize(
            rx_grid, channel_est, noise_var, data_mask
        )
        lmmse = lmmse_equalize(
            rx_grid, channel_est, normalized_n0, data_mask.any(dim=-1)
        )
        batch, symbols, subcarriers, _, _ = rx_grid.shape

        rx_features = rx_grid.permute(0, 3, 4, 1, 2).reshape(
            batch, 2 * self.num_rx_antennas, symbols, subcarriers
        )
        rx_features = rx_features[:, None].expand(-1, self.num_layers, -1, -1, -1)
        h_features = channel_est.permute(0, 4, 3, 5, 1, 2).reshape(
            batch,
            self.num_layers,
            2 * self.num_rx_antennas,
            symbols,
            subcarriers,
        )
        x_features = lmmse.permute(0, 3, 4, 1, 2)
        mask_features = data_mask.permute(0, 3, 1, 2)[:, :, None].to(rx_grid.dtype)
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
        evidence = torch.cat(
            (rx_features, h_features, x_features, mask_features, log_n0, conditions),
            dim=2,
        )
        flat_evidence = evidence.reshape(
            batch * self.num_layers, evidence.shape[2], symbols, subcarriers
        )
        state = self.state_init(flat_evidence).reshape(
            batch, self.num_layers, -1, symbols, subcarriers
        )
        for iteration in self.iterations:
            state = iteration(state, evidence)

        flat_state = state.reshape(batch * self.num_layers, -1, symbols, subcarriers)
        symbol_output = self.symbol_head(flat_state).reshape(
            batch, self.num_layers, 3, symbols, subcarriers
        )
        correction = symbol_output[:, :, :2].permute(0, 3, 4, 1, 2)
        log_symbol_var = symbol_output[:, :, 2].permute(0, 2, 3, 1).clamp(-10.0, 6.0)
        equalized = lmmse + correction * data_mask[..., None].to(correction.dtype)
        bit_logits = self.bit_head(flat_state).reshape(
            batch,
            self.num_layers,
            self.max_bits_per_symbol,
            symbols,
            subcarriers,
        ).permute(0, 3, 4, 1, 2)
        bit_logits = bit_logits * data_mask[..., None].to(bit_logits.dtype)
        return {
            "equalized_symbols": equalized,
            "bit_logits": bit_logits,
            "log_symbol_var": log_symbol_var,
            "lmmse_symbols": lmmse,
        }
