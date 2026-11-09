#!/usr/bin/env python3
"""Train the PyTorch 2x2 MIMO PDSCH neural receiver."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_rx_pytorch.checkpoint import EXPECTED_ARCHITECTURE, save_checkpoint
from neural_rx_pytorch.config import load_config, save_config
from neural_rx_pytorch.data import MatlabH5Dataset, ResourceGridBatchSampler, padded_collate
from neural_rx_pytorch.engine import run_epoch
from neural_rx_pytorch.losses import ReceiverLoss
from neural_rx_pytorch.model import NeuralPDSCHReceiver
from neural_rx_pytorch.splits import (
    load_splits,
    make_stratified_splits,
    save_splits,
    split_statistics,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True, help="HDF5 files, glob, or directory")
    parser.add_argument("--config", default=str(ROOT / "config" / "default_2x2.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "default"))
    parser.add_argument("--resume", default=None, help="Checkpoint to resume, usually last.pt")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    plotted_ber = [
        row["validation_ber"]
        if row["validation_ber"] > 0.0
        else row["validation_ber_upper_95"]
        for row in history
    ]
    axes[1].semilogy(epochs, plotted_ber, label="BER (95% upper bound if 0 errors)")
    axes[1].plot(
        epochs,
        [100.0 * row["validation_neural_evm"] for row in history],
        label="Neural EVM (%)",
    )
    axes[1].set(xlabel="Epoch", title="Validation metrics")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    epochs = args.epochs or int(training["epochs"])
    batch_size = args.batch_size or int(training["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(training["num_workers"])
    seed = int(training["seed"])
    seed_everything(seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device. Use --device cpu.")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MatlabH5Dataset(
        args.data, max_bits_per_symbol=int(config["model"]["max_bits_per_symbol"])
    )
    split = config["split"]
    split_path = output_dir / "splits.npz"
    if args.resume:
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Resume requires the original split file: {split_path}"
            )
        splits = load_splits(split_path, dataset)
    else:
        splits = make_stratified_splits(
            dataset,
            float(split["train"]),
            float(split["validation"]),
            float(split["test"]),
            seed,
        )
    empty_splits = [name for name, indices in splits.items() if len(indices) == 0]
    if empty_splits:
        raise ValueError(
            f"Dataset is too small for non-empty splits {empty_splits}. "
            "Collect more samples per (MCS, Qm, RB, SNR) group or adjust split fractions."
        )
    if not args.resume:
        save_splits(splits, dataset, split_path)
    (output_dir / "split_report.json").write_text(
        json.dumps(split_statistics(splits, dataset), indent=2), encoding="utf-8"
    )
    print(
        f"Samples: train={len(splits['train'])}, validation={len(splits['validation'])}, "
        f"test={len(splits['test'])}"
    )
    loader_options = dict(
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=padded_collate,
    )
    train_batch_sampler = ResourceGridBatchSampler(
        dataset, splits["train"], batch_size, shuffle=True, seed=seed
    )
    validation_batch_sampler = ResourceGridBatchSampler(
        dataset, splits["validation"], batch_size, shuffle=False
    )
    train_loader = DataLoader(
        dataset, batch_sampler=train_batch_sampler, **loader_options
    )
    validation_loader = DataLoader(
        dataset, batch_sampler=validation_batch_sampler, **loader_options
    )

    model = NeuralPDSCHReceiver.from_config(config["model"]).to(device)
    criterion = ReceiverLoss(**config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    amp_enabled = bool(training["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    start_epoch = 1
    best_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float]] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("architecture") != EXPECTED_ARCHITECTURE:
            raise ValueError(
                "The resume checkpoint is not a joint DMRS receiver checkpoint. "
                "Start a new run; LMMSE-assisted weights are not compatible."
            )
        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError as error:
            raise ValueError(
                "Resume checkpoint and config do not describe the same model. "
                "Use the config.json from the original run."
            ) from error
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_validation_loss"])
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        history = list(checkpoint.get("history", []))
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    save_config(config, output_dir / "config.json")

    for epoch in range(start_epoch, epochs + 1):
        train_batch_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            amp_enabled=amp_enabled,
        )
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device, amp_enabled=amp_enabled
        )
        scheduler.step(validation_metrics["loss"])
        row: dict[str, float] = {"epoch": float(epoch), "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(row)

        improved = validation_metrics["loss"] < best_loss
        if improved:
            best_loss = validation_metrics["loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        state = {
            "architecture": EXPECTED_ARCHITECTURE,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_validation_loss": best_loss,
            "stale_epochs": stale_epochs,
            "history": history,
            "config": config,
            "data_paths": [str(path) for path in dataset.paths],
        }
        save_checkpoint(state, output_dir / "last.pt")
        if improved:
            save_checkpoint(state, output_dir / "best.pt")

        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        plot_history(history, output_dir / "loss.png")
        print(
            f"Epoch {epoch:03d}/{epochs:03d} train={train_metrics['loss']:.6f} "
            f"val={validation_metrics['loss']:.6f} BER={validation_metrics['ber']:.3e} "
            f"({int(validation_metrics['bit_errors'])}/{int(validation_metrics['bit_count'])} bits; "
            f"95% upper={validation_metrics['ber_upper_95']:.3e}) "
            f"EVM={100*validation_metrics['neural_evm']:.2f}%"
        )
        if stale_epochs >= int(training["early_stop_patience"]):
            print("Early stopping: validation loss did not improve.")
            break

    dataset.close()
    print(f"Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
