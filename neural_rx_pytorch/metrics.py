"""Receiver validation metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def masked_evm(estimate: Tensor, reference: Tensor, mask: Tensor) -> Tensor:
    error_power, reference_power = masked_evm_sums(estimate, reference, mask)
    return torch.sqrt(error_power / reference_power.clamp_min(1e-12))


def masked_evm_sums(
    estimate: Tensor, reference: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor]:
    mask = mask.to(torch.float64)
    error = (estimate.double() - reference.double()).square().sum(dim=-1)
    power = reference.double().square().sum(dim=-1)
    return (error * mask).sum(), (power * mask).sum()


def masked_ber(logits: Tensor, bits: Tensor, data_mask: Tensor, qm: Tensor) -> Tensor:
    """Return coded-bit BER without inheriting AMP's float16 dtype.

    A normal PDSCH batch contains far more than 65,504 valid bits. Casting the
    counters to the logits dtype therefore overflows when CUDA autocast emits
    float16 logits, producing ``inf/inf`` (NaN) or a false zero BER.
    """

    error_count, bit_count = masked_ber_counts(logits, bits, data_mask, qm)
    return error_count / bit_count.clamp_min(1.0)


def masked_ber_counts(
    logits: Tensor, bits: Tensor, data_mask: Tensor, qm: Tensor
) -> tuple[Tensor, Tensor]:
    positions = torch.arange(logits.shape[-1], device=logits.device)
    mask = data_mask.bool()[..., None] & (positions < qm[:, None, None, None, None])
    errors = ((logits >= 0) != (bits >= 0.5)) & mask
    return errors.sum(dtype=torch.float64), mask.sum(dtype=torch.float64)


class MetricAccumulator:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.count = 0

    def update(self, values: dict[str, float], weight: int) -> None:
        for name, value in values.items():
            self.totals[name] = self.totals.get(name, 0.0) + float(value) * weight
        self.count += weight

    def compute(self) -> dict[str, float]:
        if not self.count:
            raise RuntimeError("No samples were accumulated.")
        return {name: value / self.count for name, value in self.totals.items()}


class ReceiverMetricAccumulator:
    """Accumulate exact BER and EVM numerators across variable-size batches."""

    def __init__(self) -> None:
        self.losses = MetricAccumulator()
        self.bit_errors = 0.0
        self.bit_count = 0.0
        self.neural_error_power = 0.0
        self.neural_reference_power = 0.0
        self.traditional_error_power = 0.0
        self.traditional_reference_power = 0.0
        self.traditional_samples = 0

    @property
    def count(self) -> int:
        return self.losses.count

    def update(
        self,
        losses: dict[str, Tensor],
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
        traditional: Tensor | None = None,
    ) -> None:
        batch_size = int(batch["rx_grid"].shape[0])
        self.losses.update(
            {name: float(value.detach()) for name, value in losses.items()}, batch_size
        )
        errors, bits = masked_ber_counts(
            outputs["bit_logits"], batch["coded_bits"], batch["data_mask"], batch["qm"]
        )
        neural_error, neural_reference = masked_evm_sums(
            outputs["equalized_symbols"], batch["tx_symbols"], batch["data_mask"]
        )
        self.bit_errors += float(errors)
        self.bit_count += float(bits)
        self.neural_error_power += float(neural_error)
        self.neural_reference_power += float(neural_reference)
        if traditional is not None:
            traditional_error, traditional_reference = masked_evm_sums(
                traditional, batch["tx_symbols"], batch["data_mask"]
            )
            self.traditional_error_power += float(traditional_error)
            self.traditional_reference_power += float(traditional_reference)
            self.traditional_samples += batch_size

    def compute(self) -> dict[str, float]:
        result = self.losses.compute()
        result["ber"] = self.bit_errors / max(1.0, self.bit_count)
        result["neural_evm"] = (
            self.neural_error_power / max(1e-12, self.neural_reference_power)
        ) ** 0.5
        if self.traditional_samples:
            result["traditional_evm"] = (
                self.traditional_error_power
                / max(1e-12, self.traditional_reference_power)
            ) ** 0.5
        return result
