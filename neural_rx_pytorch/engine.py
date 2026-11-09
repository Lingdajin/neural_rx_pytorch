"""Shared train and evaluation loops."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from .metrics import ReceiverMetricAccumulator


MODEL_INPUTS = (
    "rx_grid",
    "noise_var",
    "data_mask",
    "qm",
    "code_rate",
    "num_rbs",
    "dmrs_grid",
    "dmrs_mask",
)


def ablate_dmrs(batch: dict[str, Tensor], mode: str) -> dict[str, Tensor]:
    """Evaluation controls only; preserve RX, labels and the original batch."""
    if mode == "correct":
        return batch
    result = dict(batch)
    grid = batch["dmrs_grid"]
    if mode == "zero":
        result["dmrs_grid"] = torch.zeros_like(grid)
        result["dmrs_mask"] = torch.zeros_like(batch["dmrs_mask"])
    elif mode == "phase_rotate":
        result["dmrs_grid"] = torch.stack((-grid[..., 1], grid[..., 0]), dim=-1)
    else:
        raise ValueError(f"Unknown DM-RS control: {mode}")
    return result


def move_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def run_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float = 1.0,
    amp_enabled: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    accumulator = ReceiverMetricAccumulator()
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            batch = move_to_device(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled and device.type == "cuda",
            ):
                outputs = model(*(batch[name] for name in MODEL_INPUTS))
                losses = criterion(outputs, batch)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(losses["loss"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()
            with torch.no_grad():
                baseline = batch.get("traditional_eq")
                accumulator.update(losses, outputs, batch, baseline)
    return accumulator.compute()
