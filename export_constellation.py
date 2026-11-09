#!/usr/bin/env python3
"""Export neural/traditional/reference constellation points and EVM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader, Subset

from neural_rx_pytorch.checkpoint import load_model_checkpoint
from neural_rx_pytorch.config import load_config
from neural_rx_pytorch.data import MatlabH5Dataset, padded_collate
from neural_rx_pytorch.engine import MODEL_INPUTS, move_to_device
from neural_rx_pytorch.model import NeuralPDSCHReceiver
from neural_rx_pytorch.model import lmmse_equalize
from neural_rx_pytorch.splits import load_splits


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--config", default=str(ROOT / "config" / "default_2x2.json"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", default=str(ROOT / "results" / "constellation.mat"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-samples", type=int, default=100)
    return parser.parse_args()


def to_complex(value: torch.Tensor) -> np.ndarray:
    value = value.detach().cpu().numpy()
    return value[..., 0] + 1j * value[..., 1]


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu.")
    device = torch.device(args.device)
    config = load_config(args.config)
    dataset = MatlabH5Dataset(
        args.data, max_bits_per_symbol=int(config["model"]["max_bits_per_symbol"])
    )
    indices = load_splits(args.splits, dataset)[args.split][: args.max_samples]
    loader = DataLoader(
        Subset(dataset, indices.tolist()), batch_size=1, shuffle=False, collate_fn=padded_collate
    )
    model = NeuralPDSCHReceiver.from_config(config["model"]).to(device)
    load_model_checkpoint(args.checkpoint, model, device)
    fields: dict[str, list[np.ndarray]] = {
        "reference": [], "neural": [], "traditional": [], "mcs_index": [],
        "num_rbs": [], "layer": [], "sample_index": [],
    }
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = model(*(batch[name] for name in MODEL_INPUTS))
            traditional = batch.get("traditional_eq")
            if traditional is None:
                if "channel_est" not in batch:
                    raise ValueError(
                        "Constellation comparison needs traditional_eq or channel_est."
                    )
                traditional = lmmse_equalize(
                    batch["rx_grid"],
                    batch["channel_est"],
                    batch["noise_var"],
                    batch["data_mask"].any(dim=-1),
                )
            mask = batch["data_mask"][0].cpu().numpy()
            reference = to_complex(batch["tx_symbols"])[0]
            neural = to_complex(outputs["equalized_symbols"])[0]
            traditional_np = to_complex(traditional)[0]
            for layer in range(2):
                selected = mask[..., layer]
                count = int(selected.sum())
                fields["reference"].append(reference[..., layer][selected])
                fields["neural"].append(neural[..., layer][selected])
                fields["traditional"].append(traditional_np[..., layer][selected])
                fields["mcs_index"].append(np.full(count, int(batch["mcs_index"][0]), np.int32))
                fields["num_rbs"].append(np.full(count, int(batch["num_rbs"][0]), np.int32))
                fields["layer"].append(np.full(count, layer + 1, np.int32))
                fields["sample_index"].append(
                    np.full(count, int(batch["sample_index"][0]), np.int64)
                )
    exported = {name: np.concatenate(values) for name, values in fields.items()}
    ref_power = np.sum(np.abs(exported["reference"]) ** 2)
    exported["neural_evm_percent"] = np.asarray(
        [100 * np.sqrt(np.sum(np.abs(exported["neural"] - exported["reference"]) ** 2) / ref_power)]
    )
    exported["traditional_evm_percent"] = np.asarray(
        [100 * np.sqrt(np.sum(np.abs(exported["traditional"] - exported["reference"]) ** 2) / ref_power)]
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    savemat(output, exported, do_compression=True)
    np.savez_compressed(output.with_suffix(".npz"), **exported)
    dataset.close()
    print(json.dumps({
        "output": str(output), "points": len(exported["reference"]),
        "neural_evm_percent": float(exported["neural_evm_percent"][0]),
        "traditional_evm_percent": float(exported["traditional_evm_percent"][0]),
    }, indent=2))


if __name__ == "__main__":
    main()
