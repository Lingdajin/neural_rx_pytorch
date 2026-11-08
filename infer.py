#!/usr/bin/env python3
"""Run a trained receiver on a MATLAB-compatible HDF5 input file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader

from neural_rx_pytorch.checkpoint import load_model_checkpoint
from neural_rx_pytorch.config import load_config
from neural_rx_pytorch.data import MatlabH5Dataset, padded_collate
from neural_rx_pytorch.engine import MODEL_INPUTS, move_to_device
from neural_rx_pytorch.model import NeuralPDSCHReceiver


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Inference HDF5 file")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="MAT v5 result file")
    parser.add_argument("--config", default=str(ROOT / "config" / "default_2x2.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def as_complex(value: torch.Tensor) -> np.ndarray:
    value = value.detach().cpu().numpy()
    return value[..., 0] + 1j * value[..., 1]


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu.")
    device = torch.device(args.device)
    config = load_config(args.config)
    dataset = MatlabH5Dataset(
        args.input,
        max_bits_per_symbol=int(config["model"]["max_bits_per_symbol"]),
        require_targets=False,
    )
    # MATLAB calls normally contain one RB/MCS shape. Padded output is still
    # accompanied by data_mask for callers that place several samples in a file.
    loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=padded_collate)
    batch = move_to_device(next(iter(loader)), device)
    model = NeuralPDSCHReceiver.from_config(config["model"]).to(device)
    load_model_checkpoint(args.checkpoint, model, device)
    model.eval()
    with torch.no_grad():
        outputs = model(*(batch[name] for name in MODEL_INPUTS))
    result = {
        "equalized_symbols": as_complex(outputs["equalized_symbols"]),
        "bit_llr_positive_one": outputs["bit_logits"].cpu().numpy(),
        "log_symbol_variance": outputs["log_symbol_var"].cpu().numpy(),
        "data_mask": batch["data_mask"].cpu().numpy(),
        "mcs_index": batch["mcs_index"].cpu().numpy(),
        "qm": batch["qm"].cpu().numpy(),
        "num_rbs": batch["num_rbs"].cpu().numpy(),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    savemat(output, result, do_compression=True)
    dataset.close()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

