"""Deterministic, stratified train/validation/test splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .data import MatlabH5Dataset

SPLIT_SCHEMA_VERSION = 2


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
    base_groups: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(
        zip(
            dataset.metadata["mcs_index"],
            dataset.metadata["qm"],
            dataset.metadata["num_rbs"],
        )
    ):
        base_groups.setdefault(
            (int(key[0]), int(key[1]), int(key[2])), []
        ).append(index)

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
    result = {
        name: rng.permutation(np.asarray(indices, dtype=np.int64))
        for name, indices in output.items()
    }
    validate_splits(result, len(dataset))
    return result


def validate_splits(splits: dict[str, np.ndarray], dataset_length: int) -> None:
    """Reject missing, duplicated, overlapping, or out-of-range sample indices."""

    expected = {"train", "validation", "test"}
    if set(splits) != expected:
        raise ValueError(f"Split names must be {sorted(expected)}.")
    arrays = [np.asarray(splits[name], dtype=np.int64) for name in sorted(expected)]
    merged = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
    if np.any(merged < 0) or np.any(merged >= dataset_length):
        raise ValueError("Split file contains an out-of-range sample index.")
    if len(np.unique(merged)) != len(merged):
        raise ValueError("Train/validation/test splits overlap or contain duplicates.")
    if len(merged) != dataset_length:
        raise ValueError(
            f"Splits cover {len(merged)} of {dataset_length} dataset samples."
        )


def split_statistics(
    splits: dict[str, np.ndarray], dataset: MatlabH5Dataset
) -> dict[str, object]:
    """Return auditable per-scenario counts for the saved data partition."""

    validate_splits(splits, len(dataset))
    snr = dataset.metadata.get("snr_db")

    def key(index: int) -> tuple[int, int, int, float | None]:
        return (
            int(dataset.metadata["mcs_index"][index]),
            int(dataset.metadata["qm"][index]),
            int(dataset.metadata["num_rbs"][index]),
            None if snr is None else float(snr[index]),
        )

    counts = {
        name: Counter(key(int(index)) for index in indices)
        for name, indices in splits.items()
    }
    scenario_keys = sorted(
        set().union(*(counter.keys() for counter in counts.values())),
        key=lambda value: (value[0], value[1], value[2], float("-inf") if value[3] is None else value[3]),
    )
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "samples": {name: int(len(indices)) for name, indices in splits.items()},
        "scenario_count": len(scenario_keys),
        "scenarios": [
            {
                "mcs_index": scenario[0],
                "qm": scenario[1],
                "num_rbs": scenario[2],
                "snr_db": scenario[3],
                **{name: counts[name][scenario] for name in ("train", "validation", "test")},
            }
            for scenario in scenario_keys
        ],
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
        split_schema_version=np.asarray([SPLIT_SCHEMA_VERSION], dtype=np.int64),
    )


def load_splits(path: str | Path, dataset: MatlabH5Dataset) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        version = int(stored["split_schema_version"][0]) if "split_schema_version" in stored else 1
        if version != SPLIT_SCHEMA_VERSION:
            raise ValueError(
                f"Split file uses schema {version}, but schema {SPLIT_SCHEMA_VERSION} is required. "
                "Create a new output directory (or remove splits.npz) to regenerate the "
                "MCS/Qm/RB/SNR-stratified split."
            )
        expected_length = int(stored["dataset_length"][0])
        expected_fingerprint = str(stored["fingerprint"][0])
        if expected_length != len(dataset) or expected_fingerprint != dataset_fingerprint(dataset):
            raise ValueError("Split file belongs to a different dataset version.")
        result = {
            name: np.asarray(stored[name], dtype=np.int64)
            for name in ("train", "validation", "test")
        }
        validate_splits(result, len(dataset))
        return result
