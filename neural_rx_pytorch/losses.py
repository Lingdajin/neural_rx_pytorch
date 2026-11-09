"""Masked objectives for variable MCS and resource allocations."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ReceiverLoss(nn.Module):
    def __init__(
        self,
        bit_weight: float = 1.0,
        symbol_weight: float = 0.2,
        channel_weight: float = 0.05,
        pilot_weight: float = 0.05,
    ):
        super().__init__()
        self.bit_weight = bit_weight
        self.symbol_weight = symbol_weight
        self.channel_weight = channel_weight
        self.pilot_weight = pilot_weight

    def forward(
        self, outputs: dict[str, Tensor], batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        data_mask = batch["data_mask"].bool()
        max_bits = outputs["bit_logits"].shape[-1]
        bit_positions = torch.arange(max_bits, device=data_mask.device)
        bit_mask = data_mask[..., None] & (bit_positions < batch["qm"][:, None, None, None, None])

        raw_bce = F.binary_cross_entropy_with_logits(
            outputs["bit_logits"], batch["coded_bits"], reduction="none"
        )
        bit_loss = (raw_bce * bit_mask).sum() / bit_mask.sum().clamp_min(1)

        error_power = (outputs["equalized_symbols"] - batch["tx_symbols"]).square().sum(-1)
        reference_power = batch["tx_symbols"].square().sum(-1)
        symbol_loss = (error_power * data_mask).sum() / (
            (reference_power * data_mask).sum().clamp_min(1e-8)
        )
        zero = bit_loss.new_zeros(())
        channel_loss = zero
        if "channel_est" in batch and self.channel_weight > 0.0:
            channel_mask = data_mask[:, :, :, None, :, None]
            channel_error = (
                outputs["refined_channel"] - batch["channel_est"]
            ).square()
            channel_power = batch["channel_est"].square()
            channel_loss = (channel_error * channel_mask).sum() / (
                (channel_power * channel_mask).sum().clamp_min(1e-8)
            )

        rx = torch.view_as_complex(batch["rx_grid"].float().contiguous())
        channel = torch.view_as_complex(
            outputs["refined_channel"].float().contiguous()
        )
        dmrs = torch.view_as_complex(batch["dmrs_grid"].float().contiguous())
        reconstructed = torch.einsum("bskrl,bskl->bskr", channel, dmrs)
        pilot_mask = batch["dmrs_mask"].any(dim=-1)
        pilot_error = (rx - reconstructed).abs().square()
        pilot_power = rx.abs().square()
        pilot_loss = (pilot_error * pilot_mask[..., None]).sum() / (
            (pilot_power * pilot_mask[..., None]).sum().clamp_min(1e-8)
        )

        total = (
            self.bit_weight * bit_loss
            + self.symbol_weight * symbol_loss
            + self.channel_weight * channel_loss
            + self.pilot_weight * pilot_loss
        )
        return {
            "loss": total,
            "bit_loss": bit_loss,
            "symbol_loss": symbol_loss,
            "channel_loss": channel_loss,
            "pilot_loss": pilot_loss,
        }

