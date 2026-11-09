#!/usr/bin/env python3
"""Bounded small-sample training with/without known pilots; not a BLER benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neural_rx_pytorch.config import load_config
from neural_rx_pytorch.data import MatlabH5Dataset, padded_collate
from neural_rx_pytorch.engine import ablate_dmrs, move_to_device, run_epoch
from neural_rx_pytorch.losses import ReceiverLoss
from neural_rx_pytorch.model import NeuralPDSCHReceiver

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", default=[str(ROOT / "dataset_h5")])
    parser.add_argument("--config", default=str(ROOT / "config/default_2x2.json"))
    parser.add_argument("--output", default=str(ROOT / "results/dmrs_probe.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--qm", type=int, default=2)
    parser.add_argument("--num-rbs", type=int, default=10)
    parser.add_argument("--min-snr", type=float, default=20)
    parser.add_argument("--train-samples", type=int, default=16)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train_samples, args.validation_samples, args.steps, args.batch_size) < 1:
        parser.error("sample counts, steps and batch size must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output; already exists: {output}")
    config = load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu.")
    dataset = MatlabH5Dataset(args.data, config["model"]["max_bits_per_symbol"])
    try:
        if "snr_db" not in dataset.metadata:
            raise ValueError("This controlled probe requires snr_db for sample selection.")
        eligible = np.flatnonzero(
            (dataset.metadata["qm"] == args.qm)
            & (dataset.metadata["num_rbs"] == args.num_rbs)
            & (dataset.metadata["snr_db"] >= args.min_snr)
        )
        count = args.train_samples + args.validation_samples
        if len(eligible) < count:
            raise ValueError(f"Need {count} matching samples, found {len(eligible)}.")
        selected = np.random.default_rng(args.seed).choice(eligible, count, replace=False)
        samples = [dataset[int(index)] for index in selected]
        paths = [str(path) for path in dataset.paths]
    finally:
        dataset.close()

    # Cache only this small diagnostic subset. Training/validation indices are disjoint.
    train_batches = [move_to_device(padded_collate(samples[i:min(i + args.batch_size, args.train_samples)]), device)
                     for i in range(0, args.train_samples, args.batch_size)]
    validation_batches = [move_to_device(padded_collate(samples[i:i + args.batch_size]), device)
                          for i in range(args.train_samples, count, args.batch_size)]
    criterion = ReceiverLoss(**config["loss"])
    amp = device.type == "cuda" and bool(config["training"]["amp"])
    experiments = {}
    for mode in ("correct", "zero"):
        # Identical initialization, sample order, optimizer and step budget.
        torch.manual_seed(args.seed)
        model = NeuralPDSCHReceiver.from_config(config["model"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
        initial = run_epoch(model, [ablate_dmrs(batch, mode) for batch in train_batches],
                            criterion, device, amp_enabled=amp)
        for step in range(args.steps):
            batch = ablate_dmrs(train_batches[step % len(train_batches)], mode)
            metrics = run_epoch(
                model, [batch], criterion, device, optimizer, scaler,
                gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
                amp_enabled=amp,
            )
            if (step + 1) % 50 == 0 or step + 1 == args.steps:
                print(f"{mode} step={step + 1}: loss={metrics['loss']:.4f} "
                      f"EVM={100 * metrics['neural_evm']:.2f}% BER={metrics['ber']:.4f}", flush=True)
        evaluations = {}
        for split, batches in (("train", train_batches), ("validation", validation_batches)):
            evaluations[split] = {
                control: run_epoch(model, [ablate_dmrs(batch, control) for batch in batches],
                                   criterion, device, amp_enabled=amp)
                for control in ("correct", "zero", "phase_rotate")
            }
        experiments[mode] = {"initial_train": initial, "evaluation": evaluations}
        del model, optimizer, scaler
    report = {
        "architecture": NeuralPDSCHReceiver.architecture,
        "config": config, "arguments": vars(args), "data_paths": paths,
        "train_indices": selected[:args.train_samples].tolist(),
        "validation_indices": selected[args.train_samples:].tolist(),
        "experiments": experiments,
        "limitations": "Small-sample memorization/ablation probe only. Disjoint sample indices do not guarantee independent channel trajectories. Does not prove exact DM-RS alignment or full-data generalization.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
