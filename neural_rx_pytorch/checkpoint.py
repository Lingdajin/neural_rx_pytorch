"""Checkpoint IO."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

EXPECTED_ARCHITECTURE = "joint_dmrs_cgnn_v1"


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_model_checkpoint(
    path: str | Path, model: nn.Module, device: torch.device
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture != EXPECTED_ARCHITECTURE:
        raise ValueError(
            f"Checkpoint architecture is {architecture!r}; expected "
            f"{EXPECTED_ARCHITECTURE!r}. Legacy LMMSE-assisted checkpoints cannot be "
            "loaded into the joint DMRS receiver and must be retrained."
        )
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint parameters do not match the configured joint receiver. "
            "Use the config.json saved beside the checkpoint."
        ) from error
    return checkpoint

