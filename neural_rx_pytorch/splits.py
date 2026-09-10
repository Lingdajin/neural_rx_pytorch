"""Deterministic, stratified train/validation/test splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import MatlabH5Dataset


def dataset_fingerprint(dataset: MatlabH5Dataset) -> str:
    description = [
        {
            "path": str(shard.path),
            "count": shard.count,
            "size": shard.path.stat().st_size,
            "mtime_ns": shard.path.stat().st_mtime_ns,
        }
        for shard in dataset.shards
    ]
    encoded = json.dumps(description, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_stratified_splits(
    dataset: MatlabH5Dataset,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction], dtype=np.float64
    )
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Split fractions must be non-negative and sum to one.")
    rng = np.random.default_rng(seed)
    base_groups: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(
        zip(dataset.metadata["mcs_index"], dataset.metadata["num_rbs"])
    ):
        base_groups.setdefault((int(key[0]), int(key[1])), []).append(index)

    # Discrete SNR sweeps should occur in every split. Continuously sampled SNR
    # values are unsuitable exact keys, so add SNR only when every subgroup can
    # populate all requested non-empty splits.
    groups: list[list[int]] = []
    snr_values = dataset.metadata.get("snr_db")
    required_group_size = int(np.count_nonzero(fractions))
    for indices in base_groups.values():
        if snr_values is None:
            groups.append(indices)
            continue
        by_snr: dict[float, list[int]] = {}
        for index in indices:
            by_snr.setdefault(float(snr_values[index]), []).append(index)
        if len(by_snr) > 1 and min(map(len, by_snr.values())) >= required_group_size:
            groups.extend(by_snr.values())
        else:
            groups.append(indices)
    output: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for indices in groups:
        values = np.asarray(indices, dtype=np.int64)
        rng.shuffle(values)
        count = len(values)
        validation_count = int(round(count * validation_fraction))
        test_count = int(round(count * test_fraction))
        if count >= 3 and validation_fraction > 0:
            validation_count = max(1, validation_count)
        if count >= 3 and test_fraction > 0:
            test_count = max(1, test_count)
        if validation_count + test_count >= count:
            overflow = validation_count + test_count - count + 1
            test_count = max(0, test_count - overflow)
        train_stop = count - validation_count - test_count
        validation_stop = train_stop + validation_count
        output["train"].extend(values[:train_stop])
        output["validation"].extend(values[train_stop:validation_stop])
        output["test"].extend(values[validation_stop:])
    return {
        name: rng.permutation(np.asarray(indices, dtype=np.int64))
        for name, indices in output.items()
    }


def save_splits(
    splits: dict[str, np.ndarray], dataset: MatlabH5Dataset, path: str | Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **splits,
        dataset_length=np.asarray([len(dataset)], dtype=np.int64),
        fingerprint=np.asarray([dataset_fingerprint(dataset)]),
    )


def load_splits(path: str | Path, dataset: MatlabH5Dataset) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        expected_length = int(stored["dataset_length"][0])
        expected_fingerprint = str(stored["fingerprint"][0])
        if expected_length != len(dataset) or expected_fingerprint != dataset_fingerprint(dataset):
            raise ValueError("Split file belongs to a different dataset version.")
        return {
            name: np.asarray(stored[name], dtype=np.int64)
            for name in ("train", "validation", "test")
        }
