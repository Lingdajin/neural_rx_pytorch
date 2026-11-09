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
from neural_rx_pytorch.engine import MODEL_INPUTS, ablate_dmrs, run_epoch
from neural_rx_pytorch.checkpoint import check_model_architecture
from neural_rx_pytorch.losses import ReceiverLoss
from neural_rx_pytorch.metrics import masked_ber
from neural_rx_pytorch.model import NeuralPDSCHReceiver
from neural_rx_pytorch.splits import load_splits, make_stratified_splits, save_splits


def create_dataset(path: Path, axis_order: str = "python") -> None:
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
    num_rbs = np.tile(np.asarray([1, 2], np.int64), count // 2)
    pilot_mask = np.zeros_like(mask)
    pilot_mask[:, 0, ::2, :] = 1
    mask[:, 0, ::2, :] = 0
    pilot = np.zeros_like(tx)
    pilot[:, 0, ::2, 0, 0] = 1.0
    pilot[:, 0, ::2, 1, 0] = np.tile([1.0, -1.0], subcarriers // 4)
    for index, rbs in enumerate(num_rbs):
        mask[index, :, 12 * rbs :] = 0
        pilot_mask[index, :, 12 * rbs :] = 0
        pilot[index, :, 12 * rbs :] = 0
    qm = np.tile(np.asarray([2, 4], np.int64), count // 2)

    def stored(value: np.ndarray) -> np.ndarray:
        if axis_order == "matlab":
            return value.transpose(tuple(reversed(range(value.ndim))))
        return value

    with h5py.File(path, "w") as handle:
        handle.attrs["axis_order"] = axis_order
        handle.create_dataset("rx_grid", data=stored(rx))
        handle.create_dataset("channel_est", data=stored(channel))
        handle.create_dataset("data_mask", data=stored(mask))
        handle.create_dataset("dmrs_grid", data=stored(pilot))
        handle.create_dataset("dmrs_mask", data=stored(pilot_mask))
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
    def test_dmrs_validation(self) -> None:
        for corruption, message in (
            ("missing", "missing datasets"), ("shape", "logical shape"),
            ("nonbinary", "binary"), ("empty", "each layer"),
            ("overlap", "overlaps"), ("outside", "outside"),
            ("nan", "finite"), ("zero_pilot", "nonzero"),
        ):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "sample.h5"
                create_dataset(path)
                with h5py.File(path, "a") as f:
                    if corruption == "missing":
                        del f["dmrs_grid"]
                    elif corruption == "shape":
                        del f["dmrs_mask"]
                        f.create_dataset("dmrs_mask", data=np.zeros((12, 4, 24, 1)))
                    elif corruption == "nonbinary":
                        f["dmrs_mask"][0, 0, 0, 0] = 2
                    elif corruption == "empty":
                        f["dmrs_mask"][0] = 0
                    elif corruption == "overlap":
                        f["data_mask"][0, 0, 0, 0] = 1
                    elif corruption == "outside":
                        f["dmrs_grid"][0, 1, 0, 0, 0] = 1
                    elif corruption == "nan":
                        f["dmrs_grid"][0, 0, 0, 0, 0] = np.nan
                    elif corruption == "zero_pilot":
                        f["dmrs_grid"][0, 0, 0, 0] = 0
                with self.assertRaisesRegex((ValueError, KeyError), message):
                    dataset = MatlabH5Dataset(path, max_bits_per_symbol=4)
                    try:
                        dataset[0]
                    finally:
                        dataset.close()

    def test_pilot_path_and_controls(self) -> None:
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.h5"
            create_dataset(path)
            dataset = MatlabH5Dataset(path, max_bits_per_symbol=4)
            try:
                batch = padded_collate([dataset[0], dataset[1]])
            finally:
                dataset.close()
        self.assertFalse(batch["dmrs_mask"][0, :, 12:].any())
        self.assertTrue((batch["dmrs_grid"][0, :, 12:] == 0).all())
        model = NeuralPDSCHReceiver(hidden_channels=8, condition_channels=4,
                                    num_iterations=1, max_bits_per_symbol=4).eval()
        original = batch["dmrs_grid"].clone()
        batch["dmrs_grid"].requires_grad_()
        outputs = model(*(batch[name] for name in MODEL_INPUTS))
        ReceiverLoss()(outputs, batch)["loss"].backward()
        grad = batch["dmrs_grid"].grad
        self.assertTrue(torch.isfinite(grad).all())
        for layer in range(2):
            self.assertGreater(float(grad[..., layer, :].abs().sum()), 0)
        self.assertTrue((grad[~batch["dmrs_mask"]] == 0).all())
        for mode in ("zero", "phase_rotate"):
            changed = ablate_dmrs(batch, mode)
            predicted = model(*(changed[name] for name in MODEL_INPUTS))
            self.assertFalse(torch.allclose(outputs["bit_logits"], predicted["bit_logits"]))
            torch.testing.assert_close(batch["dmrs_grid"], original)
            self.assertIs(changed["rx_grid"], batch["rx_grid"])
        with self.assertRaisesRegex(ValueError, "pdsch_dmrs_v1"):
            check_model_architecture({"model_architecture": "pdsch_no_csi_v1"}, model)

    def test_selective_loading_and_channel_free_backward(self) -> None:
        for axis_order in ("python", "matlab"):
            with self.subTest(axis_order=axis_order), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "legacy.h5"
                create_dataset(path, axis_order)
                with h5py.File(path, "a") as handle:
                    # Deliberately invalid legacy fields must not even be read/validated.
                    del handle["channel_est"]
                    handle.create_dataset("channel_est", data=np.nan)
                    handle.create_dataset("traditional_eq", data=np.nan)
                read_names = []
                original_read = h5py.Dataset.__getitem__

                def tracked_read(dataset, key):
                    read_names.append(dataset.name)
                    return original_read(dataset, key)

                with patch.object(h5py.Dataset, "__getitem__", tracked_read):
                    dataset = MatlabH5Dataset(path, max_bits_per_symbol=4)
                    try:
                        batch = padded_collate([dataset[0], dataset[1]])
                    finally:
                        dataset.close()
                self.assertNotIn("/channel_est", read_names)
                self.assertNotIn("/traditional_eq", read_names)
                self.assertIn("/dmrs_grid", read_names)
                self.assertIn("/dmrs_mask", read_names)
                self.assertNotIn("channel_est", batch)
                self.assertNotIn("traditional_eq", batch)
                model = NeuralPDSCHReceiver(
                    hidden_channels=8, condition_channels=4, num_iterations=1,
                    max_bits_per_symbol=4,
                )
                outputs = model(*(batch[name] for name in MODEL_INPUTS))
                self.assertNotIn("lmmse_symbols", outputs)
                # Identical masks must not force identical spatial streams.
                self.assertFalse(torch.allclose(
                    outputs["equalized_symbols"][..., 0, :],
                    outputs["equalized_symbols"][..., 1, :],
                ))
                for name in ("equalized_symbols", "bit_logits"):
                    self.assertTrue(torch.isfinite(outputs[name]).all())
                    self.assertTrue((outputs[name][~batch["data_mask"]] == 0).all())
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                before = model.symbol_head[-1].weight.detach().clone()
                metrics = run_epoch(model, [batch], ReceiverLoss(), torch.device("cpu"), optimizer)
                self.assertNotIn("traditional_evm", metrics)
                self.assertTrue(np.isfinite(list(metrics.values())).all())
                self.assertFalse(torch.equal(before, model.symbol_head[-1].weight))
                batch["traditional_eq"] = batch["tx_symbols"].clone()
                metrics = run_epoch(model, [batch], ReceiverLoss(), torch.device("cpu"))
                self.assertEqual(metrics["traditional_evm"], 0.0)
                with self.assertRaisesRegex(ValueError, "Train a new model"):
                    check_model_architecture({"model_state": {}}, model)

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
                )
                outputs = model(*(batch[name] for name in MODEL_INPUTS))
                self.assertEqual(outputs["bit_logits"].shape, (4, 4, 24, 2, 4))
                self.assertEqual(outputs["equalized_symbols"].shape, (4, 4, 24, 2, 2))
                loss = ReceiverLoss()(outputs, batch)["loss"]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

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
                    for name in ("rx_grid", "data_mask", "dmrs_grid", "dmrs_mask", "tx_symbols", "coded_bits"):
                        torch.testing.assert_close(left[name], right[name])
            finally:
                python_data.close()
                matlab_data.close()

    def test_train_validate_export_and_infer_entrypoints(self) -> None:
        self._check_entrypoints(with_baseline=False)

    def test_entrypoints_with_stored_baseline(self) -> None:
        self._check_entrypoints(with_baseline=True)

    def _check_entrypoints(self, with_baseline: bool) -> None:
        import export_constellation
        import infer
        import train
        import validate

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "synthetic.h5"
            config_path = root / "config.json"
            run_dir = root / "run"
            create_dataset(data_path)
            with h5py.File(data_path, "a") as handle:
                del handle["channel_est"]
                if with_baseline:
                    handle.create_dataset("traditional_eq", data=handle["tx_symbols"][()])
            config = {
                "model": {
                    "num_rx_antennas": 2,
                    "num_layers": 2,
                    "max_bits_per_symbol": 4,
                    "hidden_channels": 8,
                    "condition_channels": 4,
                    "num_iterations": 1,
                    "max_num_rbs": 2,
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
                "loss": {"bit_weight": 1.0, "symbol_weight": 0.2},
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
            with patch.object(sys, "argv", train_args + ["--resume", str(run_dir / "last.pt"), "--epochs", "2"]):
                train.main()
            checkpoint = torch.load(run_dir / "last.pt", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertNotIn("train_traditional_evm", checkpoint["history"][-1])

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
            summary = json.loads((validation_dir / "summary.json").read_text())
            self.assertEqual("traditional_evm" in summary["metrics"], with_baseline)
            if with_baseline:
                self.assertEqual(summary["metrics"]["traditional_evm"], 0.0)
            self.assertEqual(summary["dmrs_mode"], "correct")
            with patch.object(sys, "argv", validation_args + ["--dmrs-mode", "phase_rotate"]):
                validate.main()
            rotated = json.loads((validation_dir / "summary.json").read_text())
            self.assertEqual(rotated["dmrs_mode"], "phase_rotate")

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
            with np.load(constellation_path.with_suffix(".npz")) as exported:
                self.assertEqual("traditional" in exported, with_baseline)

            inference_path = root / "inference.mat"
            # Deployment inputs need neither CSI nor supervised targets.
            input_path = root / "input.h5"
            create_dataset(input_path)
            with h5py.File(input_path, "a") as handle:
                for name in ("channel_est", "tx_symbols", "coded_bits"):
                    del handle[name]
            infer_args = [
                "infer.py", "--input", str(input_path), "--checkpoint",
                str(run_dir / "best.pt"), "--config", str(run_dir / "config.json"),
                "--output", str(inference_path), "--device", "cpu",
            ]
            with patch.object(sys, "argv", infer_args):
                infer.main()
            self.assertTrue(inference_path.is_file())


if __name__ == "__main__":
    unittest.main()
