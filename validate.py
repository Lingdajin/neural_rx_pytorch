#!/usr/bin/env python3
"""Evaluate a checkpoint on the saved validation or test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from neural_rx_pytorch.checkpoint import load_model_checkpoint
from neural_rx_pytorch.config import load_config
from neural_rx_pytorch.data import MatlabH5Dataset, padded_collate
from neural_rx_pytorch.engine import MODEL_INPUTS, ablate_dmrs, move_to_device
from neural_rx_pytorch.losses import ReceiverLoss
from neural_rx_pytorch.metrics import ReceiverMetricAccumulator
from neural_rx_pytorch.model import NeuralPDSCHReceiver
from neural_rx_pytorch.splits import load_splits


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--config", default=str(ROOT / "config" / "default_2x2.json"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "validation"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dmrs-mode", choices=("correct", "zero", "phase_rotate"), default="correct",
                        help="Evaluate with correct, absent, or 90-degree-rotated pilot references")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu.")
    device = torch.device(args.device)
    config = load_config(args.config)
    dataset = MatlabH5Dataset(
        args.data, max_bits_per_symbol=int(config["model"]["max_bits_per_symbol"]),
        load_traditional_eq=True,
    )
    indices = load_splits(args.splits, dataset)[args.split]
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=padded_collate,
    )
    model = NeuralPDSCHReceiver.from_config(config["model"]).to(device)
    checkpoint = load_model_checkpoint(args.checkpoint, model, device)
    criterion = ReceiverLoss(**config["loss"])

    overall = ReceiverMetricAccumulator()
    grouped: dict[tuple[int, int, float | None], ReceiverMetricAccumulator] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            batch = ablate_dmrs(batch, args.dmrs_mode)
            outputs = model(*(batch[name] for name in MODEL_INPUTS))
            for sample in range(batch["rx_grid"].shape[0]):
                selection = slice(sample, sample + 1)
                sample_batch = {name: value[selection] for name, value in batch.items()}
                sample_output = {name: value[selection] for name, value in outputs.items()}
                baseline = sample_batch.get("traditional_eq")
                sample_losses = criterion(sample_output, sample_batch)
                snr_db = (
                    float(sample_batch["snr_db"][0]) if "snr_db" in sample_batch else None
                )
                key = (
                    int(sample_batch["mcs_index"][0]),
                    int(sample_batch["num_rbs"][0]),
                    snr_db,
                )
                grouped.setdefault(key, ReceiverMetricAccumulator()).update(
                    sample_losses, sample_output, sample_batch, baseline
                )
                overall.update(sample_losses, sample_output, sample_batch, baseline)

    rows = []
    for (mcs, num_rbs, snr_db), accumulator in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or 0.0)
    ):
        row = {
            "mcs_index": mcs,
            "num_rbs": num_rbs,
            "snr_db": "" if snr_db is None else snr_db,
            "samples": accumulator.count,
        }
        row.update(accumulator.compute())
        rows.append(row)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics_by_mcs_rb.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "split": args.split,
        "dmrs_mode": args.dmrs_mode,
        "samples": len(indices),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "metrics": overall.compute(),
        "evm_baseline": "traditional_eq from dataset" if dataset.has_traditional_eq else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    dataset.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
