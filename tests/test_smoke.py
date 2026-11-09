"""Small end-to-end tests that do not depend on MATLAB or RF data."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_rx_pytorch.data import MatlabH5Dataset, padded_collate
from neural_rx_pytorch.losses import ReceiverLoss
from neural_rx_pytorch.metrics import masked_ber
from neural_rx_pytorch.model import NeuralPDSCHReceiver
from neural_rx_pytorch.splits import load_splits, make_stratified_splits, save_splits


def create_dataset(
    path: Path, axis_order: str = "python", include_channel_est: bool = True
) -> None:
    rng = np.random.default_rng(7)
    count, symbols, subcarriers = 12, 4, 24
    rx = rng.normal(size=(count, symbols, subcarriers, 2, 2)).astype(np.float32)
    channel = np.zeros((count, symbols, subcarriers, 2, 2, 2), np.float32)
    channel[..., 0, 0, 0] = 1.0
    channel[..., 1, 1, 0] = 1.0
    tx = rng.choice((-1.0, 1.0), size=(count, symbols, subcarriers, 2, 2)).astype(np.float32)
    tx /= np.sqrt(2.0)
    bits = rng.integers(0, 2, size=(count, symbols, subcarriers, 2, 4)).astype(np.float32)
    mask = np.ones((count, symbols, subcarriers, 2), np.uint8)
    dmrs = np.zeros((count, symbols, subcarriers, 2, 2), np.float32)
    dmrs_mask = np.zeros((count, symbols, subcarriers, 2), np.uint8)
    dmrs_mask[:, 1, 0::2, 0] = 1
    dmrs_mask[:, 1, 1::2, 1] = 1
    dmrs[..., 0][dmrs_mask.astype(bool)] = 1.0
    mask[dmrs_mask.astype(bool)] = 0
    num_rbs = np.tile(np.asarray([1, 2], np.int64), count // 2)
    for index, rbs in enumerate(num_rbs):
        mask[index, :, 12 * rbs :] = 0
    qm = np.tile(np.asarray([2, 4], np.int64), count // 2)

    def stored(value: np.ndarray) -> np.ndarray:
        if axis_order == "matlab":
            return value.transpose(tuple(reversed(range(value.ndim))))
        return value

    with h5py.File(path, "w") as handle:
        handle.attrs["axis_order"] = axis_order
        handle.create_dataset("rx_grid", data=stored(rx))
        handle.create_dataset("dmrs_grid", data=stored(dmrs))
        handle.create_dataset("dmrs_mask", data=stored(dmrs_mask))
        if include_channel_est:
            handle.create_dataset("channel_est", data=stored(channel))
        handle.create_dataset("data_mask", data=stored(mask))
        handle.create_dataset("tx_symbols", data=stored(tx))
        handle.create_dataset("coded_bits", data=stored(bits))
        handle.create_dataset("mcs_index", data=stored(qm))
        handle.create_dataset("qm", data=stored(qm))
        handle.create_dataset("code_rate", data=stored(np.full(count, 0.5, np.float32)))
        handle.create_dataset("num_rbs", data=stored(num_rbs))
        handle.create_dataset("noise_var", data=stored(np.full(count, 0.1, np.float32)))
        handle.create_dataset(
            "snr_db",
            data=stored(np.tile(np.asarray([0.0, 5.0], np.float32), count // 2)),
        )


class ReceiverSmokeTest(unittest.TestCase):
    def test_ber_counter_does_not_overflow_float16(self) -> None:
        bit_count = 100_000
        bits = torch.zeros((1, 1, bit_count, 1, 1), dtype=torch.float32)
        logits = torch.ones_like(bits, dtype=torch.float16)
        mask = torch.ones((1, 1, bit_count, 1), dtype=torch.bool)
        qm = torch.ones(1, dtype=torch.long)
        self.assertEqual(float(masked_ber(logits, bits, mask, qm)), 1.0)

    def test_data_model_loss_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "synthetic.h5"
            split_path = root / "splits.npz"
            create_dataset(data_path)
            dataset = MatlabH5Dataset(data_path, max_bits_per_symbol=4)
            try:
                loader = DataLoader(dataset, batch_size=4, collate_fn=padded_collate)
                batch = next(iter(loader))
                self.assertEqual(batch["rx_grid"].shape, (4, 4, 24, 2, 2))
                model = NeuralPDSCHReceiver(
                    max_bits_per_symbol=4,
                    hidden_channels=8,
                    condition_channels=4,
                    num_iterations=2,
                    max_num_rbs=2,
                    supported_qm=[2, 4],
                )
                outputs = model(
                    batch["rx_grid"], batch["dmrs_grid"], batch["dmrs_mask"],
                    batch["noise_var"], batch["data_mask"], batch["qm"],
                    batch["code_rate"], batch["num_rbs"],
                )
                self.assertEqual(outputs["bit_logits"].shape, (4, 4, 24, 2, 4))
                self.assertEqual(outputs["equalized_symbols"].shape, (4, 4, 24, 2, 2))
                loss = ReceiverLoss()(outputs, batch)["loss"]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
                no_teacher_batch = {
                    name: value for name, value in batch.items() if name != "channel_est"
                }
                self.assertTrue(
                    torch.isfinite(ReceiverLoss()(outputs, no_teacher_batch)["loss"])
                )

                splits = make_stratified_splits(dataset, 0.5, 0.25, 0.25, seed=42)
                self.assertEqual(sum(len(value) for value in splits.values()), len(dataset))
                for split_indices in splits.values():
                    self.assertEqual(
                        set(dataset.metadata["snr_db"][split_indices]), {0.0, 5.0}
                    )
                save_splits(splits, dataset, split_path)
                restored = load_splits(split_path, dataset)
                for name in splits:
                    np.testing.assert_array_equal(splits[name], restored[name])
            finally:
                dataset.close()

    def test_matlab_storage_axis_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python_path = root / "python.h5"
            matlab_path = root / "matlab.h5"
            create_dataset(python_path, "python")
            create_dataset(matlab_path, "matlab")
            python_data = MatlabH5Dataset(python_path, max_bits_per_symbol=4)
            matlab_data = MatlabH5Dataset(matlab_path, max_bits_per_symbol=4)
            try:
                for index in (0, 1, len(python_data) - 1):
                    left = python_data[index]
                    right = matlab_data[index]
                    for name in (
                        "rx_grid", "dmrs_grid", "dmrs_mask", "channel_est",
                        "data_mask", "tx_symbols", "coded_bits",
                    ):
                        torch.testing.assert_close(left[name], right[name])
            finally:
                python_data.close()
                matlab_data.close()

    def test_train_validate_export_and_infer_entrypoints(self) -> None:
        import export_constellation
        import infer
        import train
        import validate

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "synthetic.h5"
            inference_data_path = root / "synthetic_inference.h5"
            config_path = root / "config.json"
            run_dir = root / "run"
            create_dataset(data_path)
            create_dataset(inference_data_path, include_channel_est=False)
            config = {
                "model": {
                    "num_rx_antennas": 2,
                    "num_layers": 2,
                    "max_bits_per_symbol": 4,
                    "hidden_channels": 8,
                    "condition_channels": 4,
                    "pilot_channels": 4,
                    "num_iterations": 1,
                    "max_num_rbs": 2,
                    "supported_qm": [2, 4],
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 4,
                    "num_workers": 0,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "gradient_clip_norm": 1.0,
                    "early_stop_patience": 2,
                    "amp": False,
                    "seed": 42,
                },
                "split": {"train": 0.5, "validation": 0.25, "test": 0.25},
                "loss": {
                    "bit_weight": 1.0, "symbol_weight": 0.2,
                    "channel_weight": 0.05, "pilot_weight": 0.05,
                },
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            train_args = [
                "train.py", "--data", str(data_path), "--config", str(config_path),
                "--output-dir", str(run_dir), "--device", "cpu",
            ]
            with patch.object(sys, "argv", train_args):
                train.main()
            self.assertTrue((run_dir / "best.pt").is_file())
            self.assertTrue((run_dir / "loss.png").is_file())
            self.assertTrue((run_dir / "split_report.json").is_file())

            validation_dir = root / "validation"
            validation_args = [
                "validate.py", "--data", str(data_path), "--checkpoint",
                str(run_dir / "best.pt"), "--splits", str(run_dir / "splits.npz"),
                "--config", str(run_dir / "config.json"), "--output-dir",
                str(validation_dir), "--device", "cpu", "--num-workers", "0",
            ]
            with patch.object(sys, "argv", validation_args):
                validate.main()
            self.assertTrue((validation_dir / "summary.json").is_file())
            self.assertTrue((validation_dir / "metrics_by_scenario.csv").is_file())

            constellation_path = root / "constellation.mat"
            export_args = [
                "export_constellation.py", "--data", str(data_path), "--checkpoint",
                str(run_dir / "best.pt"), "--splits", str(run_dir / "splits.npz"),
                "--config", str(run_dir / "config.json"), "--output",
                str(constellation_path), "--device", "cpu", "--max-samples", "2",
            ]
            with patch.object(sys, "argv", export_args):
                export_constellation.main()
            self.assertTrue(constellation_path.is_file())

            inference_path = root / "inference.mat"
            infer_args = [
                "infer.py", "--input", str(inference_data_path), "--checkpoint",
                str(run_dir / "best.pt"), "--config", str(run_dir / "config.json"),
                "--output", str(inference_path), "--device", "cpu",
            ]
            with patch.object(sys, "argv", infer_args):
                infer.main()
            self.assertTrue(inference_path.is_file())


if __name__ == "__main__":
    unittest.main()
