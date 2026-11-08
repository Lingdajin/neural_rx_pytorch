"""Checkpoint IO."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


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
    check_model_architecture(checkpoint, model)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def check_model_architecture(checkpoint: dict[str, Any], model: nn.Module) -> None:
    if checkpoint.get("model_architecture") != model.architecture:
        raise ValueError(
            "Checkpoint architecture is incompatible with the receiver without channel "
            "estimates/LMMSE. Train a new model in a new output directory; old "
            "checkpoints cannot be resumed or used for inference."
        )

