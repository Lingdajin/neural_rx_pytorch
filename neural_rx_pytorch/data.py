"""Streaming HDF5 dataset written by MATLAB or Python."""

from __future__ import annotations

import bisect
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


ARRAY_FIELDS = {
    "rx_grid": 5,
    "data_mask": 4,
    "dmrs_grid": 5,
    "dmrs_mask": 4,
    "tx_symbols": 5,
    "coded_bits": 5,
    "traditional_eq": 5,
}
METADATA_FIELDS = ("mcs_index", "qm", "code_rate", "num_rbs", "noise_var")
OPTIONAL_METADATA_FIELDS = ("snr_db",)


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    count: int
    axis_order: str
    num_symbols: int
    max_subcarriers: int


def resolve_h5_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    resolved: list[Path] = []
    for item in paths:
        item = Path(item)
        if item.is_dir():
            matches = sorted(item.glob("*.h5")) + sorted(item.glob("*.hdf5"))
        elif any(character in str(item) for character in "*?["):
            matches = [Path(match) for match in sorted(glob.glob(str(item)))]
        else:
            matches = [item]
        resolved.extend(match.resolve() for match in matches)
    resolved = sorted(set(resolved))
    if not resolved:
        raise FileNotFoundError(f"No HDF5 dataset shards found for {paths!r}")
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return resolved


def _text_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _text_attr(value.reshape(-1)[0])
    return str(value)


def _logical_shape(dataset: h5py.Dataset, axis_order: str) -> tuple[int, ...]:
    return tuple(reversed(dataset.shape)) if axis_order == "matlab" else dataset.shape


class MatlabH5Dataset(Dataset[dict[str, Tensor]]):
    """Map-style dataset with lazy per-worker HDF5 handles.

    Files with ``axis_order=matlab`` store the sample dimension physically last;
    files with ``axis_order=python`` store it first. Returned arrays always use
    the logical sample-first convention documented in ``DATA_FORMAT.md``.
    ``channel_est`` is never opened. ``traditional_eq`` is read only when
    explicitly requested for offline evaluation; training defaults to skipping it.
    """

    def __init__(
        self,
        paths: str | Path | Iterable[str | Path],
        max_bits_per_symbol: int = 10,
        require_targets: bool = True,
        load_traditional_eq: bool = False,
    ) -> None:
        self.paths = resolve_h5_paths(paths)
        self.max_bits_per_symbol = max_bits_per_symbol
        self.require_targets = require_targets
        self.shards: list[ShardInfo] = []
        self._handles: dict[Path, h5py.File] = {}
        traditional_flags: list[bool] = []
        metadata: dict[str, list[np.ndarray]] = {name: [] for name in METADATA_FIELDS}
        optional_metadata: dict[str, list[np.ndarray]] = {
            name: [] for name in OPTIONAL_METADATA_FIELDS
        }
        optional_flags: dict[str, list[bool]] = {
            name: [] for name in OPTIONAL_METADATA_FIELDS
        }

        for path in self.paths:
            with h5py.File(path, "r") as handle:
                axis_order = _text_attr(handle.attrs.get("axis_order", "matlab")).lower()
                if axis_order not in {"matlab", "python"}:
                    raise ValueError(f"{path}: axis_order must be 'matlab' or 'python'.")
                required = {"rx_grid", "data_mask", "dmrs_grid", "dmrs_mask", *METADATA_FIELDS}
                if require_targets:
                    required.update({"tx_symbols", "coded_bits"})
                missing = sorted(required.difference(handle.keys()))
                if missing:
                    raise KeyError(f"{path}: missing datasets {missing}")
                for name in required.intersection(ARRAY_FIELDS):
                    shape = _logical_shape(handle[name], axis_order)
                    if len(shape) != ARRAY_FIELDS[name]:
                        raise ValueError(
                            f"{path}:{name} has logical shape {shape}, expected rank "
                            f"{ARRAY_FIELDS[name]}."
                        )
                rx_shape = _logical_shape(handle["rx_grid"], axis_order)
                count, num_symbols, max_subcarriers, num_rx, iq = rx_shape
                if num_rx != 2 or iq != 2:
                    raise ValueError(f"{path}: rx_grid must describe two RX antennas and IQ=2.")
                mask_shape = _logical_shape(handle["data_mask"], axis_order)
                if mask_shape != (count, num_symbols, max_subcarriers, 2):
                    raise ValueError(f"{path}: data_mask has invalid logical shape {mask_shape}.")
                for name, expected in (
                    ("dmrs_grid", (*mask_shape, 2)), ("dmrs_mask", mask_shape)
                ):
                    shape = _logical_shape(handle[name], axis_order)
                    if shape != expected:
                        raise ValueError(f"{path}: {name} has logical shape {shape}, expected {expected}.")
                if require_targets:
                    symbol_shape = _logical_shape(handle["tx_symbols"], axis_order)
                    bit_shape = _logical_shape(handle["coded_bits"], axis_order)
                    if symbol_shape != (count, num_symbols, max_subcarriers, 2, 2):
                        raise ValueError(f"{path}: tx_symbols has invalid logical shape {symbol_shape}.")
                    if bit_shape[:4] != (count, num_symbols, max_subcarriers, 2):
                        raise ValueError(f"{path}: coded_bits has invalid logical shape {bit_shape}.")
                    if bit_shape[-1] > max_bits_per_symbol:
                        raise ValueError(
                            f"{path}: coded_bits contains {bit_shape[-1]} bits/symbol, "
                            f"model limit is {max_bits_per_symbol}."
                        )
                for name in METADATA_FIELDS:
                    values = np.asarray(handle[name][()]).reshape(-1)
                    if len(values) == 1:
                        values = np.repeat(values, count)
                    if len(values) != count:
                        raise ValueError(f"{path}:{name} must contain one value per sample.")
                    metadata[name].append(values)
                for name in OPTIONAL_METADATA_FIELDS:
                    present = name in handle
                    optional_flags[name].append(present)
                    if present:
                        values = np.asarray(handle[name][()]).reshape(-1)
                        if len(values) == 1:
                            values = np.repeat(values, count)
                        if len(values) != count:
                            raise ValueError(f"{path}:{name} must contain one value per sample.")
                        optional_metadata[name].append(values)
                traditional_flags.append(load_traditional_eq and "traditional_eq" in handle)
                self.shards.append(
                    ShardInfo(path, count, axis_order, num_symbols, max_subcarriers)
                )

        if any(traditional_flags) and not all(traditional_flags):
            raise ValueError(
                "traditional_eq must be present in every shard or omitted from every shard."
            )
        self.has_traditional_eq = all(traditional_flags)

        self.offsets = np.concatenate(
            ([0], np.cumsum([shard.count for shard in self.shards], dtype=np.int64))
        )
        self.metadata = {
            name: np.concatenate(parts).astype(
                np.float32 if name in {"code_rate", "noise_var"} else np.int64
            )
            for name, parts in metadata.items()
        }
        for name in OPTIONAL_METADATA_FIELDS:
            if any(optional_flags[name]) and not all(optional_flags[name]):
                raise ValueError(f"{name} must be present in every shard or omitted from every shard.")
            if all(optional_flags[name]):
                self.metadata[name] = np.concatenate(optional_metadata[name]).astype(np.float32)
        if np.any(self.metadata["qm"] > max_bits_per_symbol):
            raise ValueError("Dataset Qm exceeds max_bits_per_symbol.")
        if np.any(self.metadata["num_rbs"] * 12 > np.repeat(
            [shard.max_subcarriers for shard in self.shards],
            [shard.count for shard in self.shards],
        )):
            raise ValueError("num_rbs exceeds the stored subcarrier dimension.")

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _handle(self, path: Path) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r", swmr=True)
        return self._handles[path]

    @staticmethod
    def _read_array(
        dataset: h5py.Dataset, sample_index: int, axis_order: str
    ) -> np.ndarray:
        if axis_order == "python":
            return np.asarray(dataset[sample_index])
        stored = np.asarray(dataset[..., sample_index])
        return stored.transpose(tuple(reversed(range(stored.ndim))))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.offsets, index) - 1
        shard = self.shards[shard_index]
        local_index = int(index - self.offsets[shard_index])
        handle = self._handle(shard.path)
        num_subcarriers = int(self.metadata["num_rbs"][index]) * 12

        result: dict[str, Tensor] = {}
        names = ["rx_grid", "data_mask", "dmrs_grid", "dmrs_mask"]
        if self.require_targets:
            names.extend(("tx_symbols", "coded_bits"))
        if self.has_traditional_eq:
            names.append("traditional_eq")
        for name in names:
            value = self._read_array(handle[name], local_index, shard.axis_order)
            value = np.ascontiguousarray(value[:, :num_subcarriers])
            if name == "dmrs_mask" and not np.all((value == 0) | (value == 1)):
                raise ValueError(f"{shard.path}: sample {local_index}: dmrs_mask must be binary.")
            if name in {"data_mask", "dmrs_mask"}:
                result[name] = torch.from_numpy(value.astype(np.bool_))
            else:
                result[name] = torch.from_numpy(value.astype(np.float32))

        pilot = result["dmrs_grid"].numpy()
        pilot_mask = result["dmrs_mask"].numpy()
        prefix = f"{shard.path}: sample {local_index}: "
        if not np.isfinite(pilot).all():
            raise ValueError(prefix + "dmrs_grid must be finite.")
        if not pilot_mask.any(axis=(0, 1)).all():
            raise ValueError(prefix + "each layer must contain known DM-RS.")
        if np.any(pilot_mask & result["data_mask"].numpy()):
            raise ValueError(prefix + "dmrs_mask overlaps data_mask on the same layer.")
        if np.any(pilot[~pilot_mask] != 0):
            raise ValueError(prefix + "dmrs_grid must be zero outside dmrs_mask.")
        if np.any(np.sum(pilot[pilot_mask].astype(np.float64) ** 2, axis=-1) == 0):
            raise ValueError(prefix + "dmrs_grid must be nonzero on dmrs_mask.")

        bit_width = result.get("coded_bits", torch.empty(0)).shape[-1]
        if self.require_targets and bit_width < self.max_bits_per_symbol:
            padding = self.max_bits_per_symbol - bit_width
            result["coded_bits"] = torch.nn.functional.pad(
                result["coded_bits"], (0, padding)
            )
        for name in METADATA_FIELDS:
            dtype = torch.float32 if name in {"code_rate", "noise_var"} else torch.long
            result[name] = torch.tensor(self.metadata[name][index], dtype=dtype)
        for name in OPTIONAL_METADATA_FIELDS:
            if name in self.metadata:
                result[name] = torch.tensor(self.metadata[name][index], dtype=torch.float32)
        result["sample_index"] = torch.tensor(index, dtype=torch.long)
        return result


def padded_collate(samples: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad the OFDM-symbol/subcarrier axes and stack a variable-RB batch."""

    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    max_symbols = max(int(sample["rx_grid"].shape[0]) for sample in samples)
    max_subcarriers = max(int(sample["rx_grid"].shape[1]) for sample in samples)
    result: dict[str, Tensor] = {}
    array_names = set(ARRAY_FIELDS).intersection(samples[0])
    for name in array_names:
        tail = samples[0][name].shape[2:]
        dtype = samples[0][name].dtype
        padded = torch.zeros(
            (len(samples), max_symbols, max_subcarriers, *tail), dtype=dtype
        )
        for position, sample in enumerate(samples):
            value = sample[name]
            padded[position, : value.shape[0], : value.shape[1]] = value
        result[name] = padded
    for name in (*METADATA_FIELDS, *OPTIONAL_METADATA_FIELDS, "sample_index"):
        if name in samples[0]:
            result[name] = torch.stack([sample[name] for sample in samples])
    return result
