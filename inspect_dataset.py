#!/usr/bin/env python3
"""Audit MATLAB HDF5 captures before neural-receiver training."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from neural_rx_pytorch.data import MatlabH5Dataset


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--config", default=str(ROOT / "config" / "default_2x2.json"))
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Inspect at most this many evenly spaced samples; default checks all.",
    )
    return parser.parse_args()


def complex_array(iq: np.ndarray) -> np.ndarray:
    return iq[..., 0].astype(np.float64) + 1j * iq[..., 1].astype(np.float64)


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    max_bits = int(config["model"]["max_bits_per_symbol"])
    dataset = MatlabH5Dataset(args.data, max_bits_per_symbol=max_bits)
    if args.max_samples is None or args.max_samples >= len(dataset):
        indices = np.arange(len(dataset), dtype=np.int64)
    else:
        indices = np.unique(
            np.linspace(0, len(dataset) - 1, args.max_samples, dtype=np.int64)
        )

    errors: list[str] = []
    warnings: list[str] = []
    scenarios: Counter[tuple[int, int, int, float]] = Counter()
    bit_counts = np.zeros(2, dtype=np.int64)
    active_bit_count = 0
    valid_re_count = 0
    total_re_count = 0
    reference_energy = 0.0
    traditional_error_energy = 0.0
    traditional_reference_energy = 0.0
    reconstruction_residual_energy = 0.0
    reconstructed_signal_energy = 0.0
    observed_rx_count = 0
    noise_variance_sum = 0.0
    reconstruction_by_snr: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])
    constellation_maps: dict[
        int, dict[tuple[int, ...], set[tuple[float, float]]]
    ] = defaultdict(lambda: defaultdict(set))
    reverse_constellation_maps: dict[
        int, dict[tuple[float, float], set[tuple[int, ...]]]
    ] = defaultdict(lambda: defaultdict(set))
    ranges: dict[str, list[float]] = defaultdict(lambda: [float("inf"), float("-inf")])

    try:
        for index in indices:
            sample = dataset[int(index)]
            qm = int(sample["qm"])
            mcs = int(sample["mcs_index"])
            num_rbs = int(sample["num_rbs"])
            snr = float(sample["snr_db"]) if "snr_db" in sample else float("nan")
            scenarios[(mcs, qm, num_rbs, snr)] += 1
            if qm not in (2, 4, 6, 8, 10):
                errors.append(f"sample {index}: unsupported Qm={qm}")
            if not 1 <= num_rbs <= int(config["model"]["max_num_rbs"]):
                errors.append(f"sample {index}: invalid num_rbs={num_rbs}")
            if not 0.0 < float(sample["code_rate"]) <= 1.0:
                errors.append(f"sample {index}: code_rate must be in (0,1]")
            if not np.isfinite(float(sample["noise_var"])) or float(sample["noise_var"]) <= 0:
                errors.append(f"sample {index}: noise_var must be finite and positive")

            array_names = ["rx_grid", "dmrs_grid", "tx_symbols", "coded_bits"]
            if "channel_est" in sample:
                array_names.append("channel_est")
            arrays = {name: sample[name].numpy() for name in array_names}
            for name, value in arrays.items():
                if not np.all(np.isfinite(value)):
                    errors.append(f"sample {index}: {name} contains NaN or Inf")
                ranges[name][0] = min(ranges[name][0], float(np.nanmin(value)))
                ranges[name][1] = max(ranges[name][1], float(np.nanmax(value)))

            mask = sample["data_mask"].numpy().astype(bool)
            dmrs_mask = sample["dmrs_mask"].numpy().astype(bool)
            dmrs = complex_array(arrays["dmrs_grid"])
            if np.any(mask & dmrs_mask):
                errors.append(f"sample {index}: data_mask and dmrs_mask overlap")
            if np.any(dmrs[~dmrs_mask] != 0.0):
                errors.append(f"sample {index}: dmrs_grid is non-zero outside dmrs_mask")
            for layer in range(2):
                layer_pilots = dmrs_mask[..., layer]
                if not layer_pilots.any():
                    errors.append(f"sample {index}: layer {layer + 1} has no DM-RS")
                elif np.any(np.abs(dmrs[..., layer][layer_pilots]) < 1e-8):
                    errors.append(
                        f"sample {index}: layer {layer + 1} has zero-valued active DM-RS"
                    )
            bits = arrays["coded_bits"]
            if not mask.any():
                errors.append(f"sample {index}: data_mask has no active RE")
                continue
            total_re_count += mask.size
            valid_re_count += int(mask.sum())
            active_bits = bits[..., :qm][np.broadcast_to(mask[..., None], bits[..., :qm].shape)]
            invalid_bits = active_bits[(active_bits != 0.0) & (active_bits != 1.0)]
            if invalid_bits.size:
                errors.append(
                    f"sample {index}: coded_bits has {invalid_bits.size} non-binary active values"
                )
            zeros = int(np.count_nonzero(active_bits == 0.0))
            ones = int(np.count_nonzero(active_bits == 1.0))
            bit_counts += (zeros, ones)
            active_bit_count += active_bits.size

            if bits.shape[-1] > qm and np.any(bits[..., qm:][mask] != 0.0):
                warnings.append(f"sample {index}: padded coded_bits beyond Qm are non-zero")
            tx = complex_array(arrays["tx_symbols"])
            reference_energy += float(np.sum(np.abs(tx[mask]) ** 2))
            labels = bits[mask][..., :qm].astype(np.int8)
            points = tx[mask]
            joined = np.column_stack(
                (labels, np.round(points.real, 6), np.round(points.imag, 6))
            )
            for row in np.unique(joined, axis=0):
                label = tuple(int(value) for value in row[:qm])
                point = (float(row[qm]), float(row[qm + 1]))
                constellation_maps[qm][label].add(point)
                reverse_constellation_maps[qm][point].add(label)
            rx = complex_array(arrays["rx_grid"])
            common_mask = mask.any(axis=-1)
            rx_values = int(np.count_nonzero(common_mask)) * rx.shape[-1]
            observed_rx_count += rx_values
            noise_variance_sum += float(sample["noise_var"]) * rx_values
            if "channel_est" in arrays:
                channel = complex_array(arrays["channel_est"])
                reconstructed = np.einsum("skrl,skl->skr", channel, tx)
                residual_energy = float(
                    np.sum(np.abs(rx[common_mask] - reconstructed[common_mask]) ** 2)
                )
                signal_energy = float(np.sum(np.abs(reconstructed[common_mask]) ** 2))
                reconstruction_residual_energy += residual_energy
                reconstructed_signal_energy += signal_energy
                if np.isfinite(snr):
                    reconstruction_by_snr[snr][0] += residual_energy
                    reconstruction_by_snr[snr][1] += signal_energy
            if np.any(tx[~mask] != 0.0):
                warnings.append(f"sample {index}: tx_symbols outside data_mask are non-zero")
            if "traditional_eq" in sample:
                traditional = complex_array(sample["traditional_eq"].numpy())
                if not np.all(np.isfinite(traditional)):
                    errors.append(f"sample {index}: traditional_eq contains NaN or Inf")
                traditional_error_energy += float(np.sum(np.abs(traditional[mask] - tx[mask]) ** 2))
                traditional_reference_energy += float(np.sum(np.abs(tx[mask]) ** 2))
    finally:
        dataset.close()

    if active_bit_count == 0:
        errors.append("No active coded bits were found")
    elif min(bit_counts) == 0:
        errors.append(f"Only one bit class is present: zeros={bit_counts[0]}, ones={bit_counts[1]}")
    bit_one_fraction = float(bit_counts[1] / max(1, bit_counts.sum()))
    if active_bit_count and not 0.4 <= bit_one_fraction <= 0.6:
        warnings.append(f"bit-one fraction {bit_one_fraction:.4f} is unexpectedly imbalanced")
    rms_symbol = float(np.sqrt(reference_energy / max(1, valid_re_count)))
    if not 0.5 <= rms_symbol <= 2.0:
        warnings.append(
            f"reference symbol RMS={rms_symbol:.4g}; confirm QAM normalization and signal scaling"
        )

    constellation_summary = {}
    for qm, mapping in sorted(constellation_maps.items()):
        reverse_mapping = reverse_constellation_maps[qm]
        ambiguous_labels = sum(len(points) != 1 for points in mapping.values())
        ambiguous_points = sum(len(labels) != 1 for labels in reverse_mapping.values())
        constellation_summary[str(qm)] = {
            "observed_bit_labels": len(mapping),
            "expected_bit_labels": 2**qm,
            "observed_points": len(reverse_mapping),
            "ambiguous_bit_labels": ambiguous_labels,
            "ambiguous_points": ambiguous_points,
        }
        if ambiguous_labels or ambiguous_points:
            errors.append(
                f"Qm={qm}: coded_bits and tx_symbols are not a one-to-one mapping"
            )
        if len(mapping) != 2**qm:
            warnings.append(
                f"Qm={qm}: observed {len(mapping)} of {2**qm} constellation labels"
            )

    unique_mcs = sorted({key[0] for key in scenarios})
    unique_rbs = sorted({key[2] for key in scenarios})
    if len(unique_mcs) < 2:
        warnings.append(
            "Only one MCS is present; this dataset cannot validate multi-MCS adaptation"
        )
    if len(unique_rbs) < 2:
        warnings.append(
            "Only one RB allocation is present; this dataset cannot validate variable-RB adaptation"
        )
    if len(dataset) < 1000:
        warnings.append(
            "Fewer than 1000 samples are present; use this capture for pipeline checks, not final performance claims"
        )

    scenario_rows = [
        {
            "mcs_index": key[0],
            "qm": key[1],
            "num_rbs": key[2],
            "snr_db": None if np.isnan(key[3]) else key[3],
            "inspected_samples": count,
        }
        for key, count in sorted(
            scenarios.items(), key=lambda item: (item[0][0], item[0][2], item[0][3])
        )
    ]
    reconstruction_snr_db = (
        10.0 * np.log10(
            reconstructed_signal_energy / max(1e-12, reconstruction_residual_energy)
        )
        if reconstructed_signal_energy > 0.0
        else None
    )
    reconstruction_rows = [
        {
            "snr_db": snr,
            "reconstructed_signal_to_residual_db": 10.0
            * np.log10(values[1] / max(1e-12, values[0])),
        }
        for snr, values in sorted(reconstruction_by_snr.items())
    ]
    report = {
        "status": "FAIL" if errors else ("WARN" if warnings else "PASS"),
        "dataset_samples": len(dataset),
        "inspected_samples": len(indices),
        "shards": len(dataset.shards),
        "has_channel_est": dataset.has_channel_est,
        "has_traditional_eq": dataset.has_traditional_eq,
        "scenarios": scenario_rows,
        "valid_re_fraction": valid_re_count / max(1, total_re_count),
        "active_bit_count": int(active_bit_count),
        "bit_zero_count": int(bit_counts[0]),
        "bit_one_count": int(bit_counts[1]),
        "bit_one_fraction": bit_one_fraction,
        "reference_symbol_rms": rms_symbol,
        "constellation_mapping": constellation_summary,
        "traditional_evm_percent": (
            100.0 * np.sqrt(traditional_error_energy / traditional_reference_energy)
            if traditional_reference_energy > 0
            else None
        ),
        "mean_declared_noise_var": noise_variance_sum / max(1, observed_rx_count),
        "mean_reconstruction_residual_power": (
            reconstruction_residual_energy / max(1, observed_rx_count)
        ),
        "reconstructed_signal_to_residual_db": reconstruction_snr_db,
        "reconstruction_by_snr": reconstruction_rows,
        "value_ranges": dict(ranges),
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    print(rendered)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
