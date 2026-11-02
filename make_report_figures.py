#!/usr/bin/env python3
"""Generate all figures for the neural receiver design report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report_figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    }
)

# 中文字体支持:优先 Microsoft YaHei,其次 SimHei;并禁用 unicode 负号。
from matplotlib import font_manager

for _font in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi"):
    if any(f.name == _font for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        print(f"using CJK font: {_font}")
        break
plt.rcParams["axes.unicode_minus"] = False


def fig_training() -> None:
    """Training curves from runs/pdsch_2x2/history.csv."""
    rows = []
    with (ROOT / "runs" / "pdsch_2x2" / "history.csv").open() as handle:
        for row in csv.DictReader(handle):
            rows.append({k: float(v) for k, v in row.items()})
    epochs = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))

    ax = axes[0, 0]
    ax.plot(epochs, [r["train_loss"] for r in rows], label="train")
    ax.plot(epochs, [r["validation_loss"] for r in rows], label="validation")
    ax.set(xlabel="Epoch", ylabel="Total loss", title="(a) 训练/验证总损失")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, [r["train_bit_loss"] for r in rows], label="train BCE")
    ax.plot(epochs, [r["validation_bit_loss"] for r in rows], label="validation BCE")
    ax.plot(epochs, [r["train_symbol_loss"] for r in rows], label="train symbol")
    ax.plot(epochs, [r["validation_symbol_loss"] for r in rows], label="validation symbol")
    ax.set(xlabel="Epoch", ylabel="Component loss", title="(b) bit BCE 与符号损失")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.semilogy(epochs, [r["validation_ber"] for r in rows], label="validation coded BER")
    ax.semilogy(epochs, [r["train_ber"] for r in rows], label="train coded BER", alpha=0.6)
    ax.set(xlabel="Epoch", ylabel="Coded BER", title="(c) 验证集 coded BER(对数坐标)")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(
        epochs,
        [100 * r["validation_neural_evm"] for r in rows],
        label="neural EVM (%)",
    )
    ax.plot(
        epochs,
        [100 * r["validation_traditional_evm"] for r in rows],
        label="traditional EVM (%)",
        alpha=0.8,
    )
    ax.set(xlabel="Epoch", ylabel="EVM (%)", title="(d) 验证集 EVM:神经 vs 传统")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("神经接收机训练过程(runs/pdsch_2x2)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig_training.png")
    plt.close(fig)
    print("wrote fig_training.png")


def fig_ber_snr() -> None:
    """Coded BER vs SNR grouped by RB from results/test/metrics_by_mcs_rb.csv."""
    rows = []
    with (ROOT / "results" / "test" / "metrics_by_mcs_rb.csv").open() as handle:
        for row in csv.DictReader(handle):
            rows.append({k: float(v) for k, v in row.items()})
    # RB adaptation must hold the modulation fixed.  MCS 6 is the QPSK
    # scenario used by the report; mixing MCS values here would create
    # multiple curves with the same RB label and invalidate the comparison.
    target_mcs = 6
    rows = [r for r in rows if int(r["mcs_index"]) == target_mcs]
    rbs = sorted({int(r["num_rbs"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for rb in rbs:
        subset = [r for r in rows if int(r["num_rbs"]) == rb]
        subset.sort(key=lambda r: r["snr_db"])
        ax.semilogy(
            [r["snr_db"] for r in subset],
            [r["ber"] for r in subset],
            marker="o",
            ms=4,
            label=f"{rb} RB",
        )
    ax.set(
        xlabel="SNR (dB)",
        ylabel="Coded BER",
        title="测试集 coded BER vs SNR(MCS 6 / QPSK, RB 自适应)",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="RB 分配")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ber_snr.png")
    plt.close(fig)
    print("wrote fig_ber_snr.png")


def fig_modulation_ber_snr() -> None:
    """Coded BER vs SNR grouped by modulation at a common RB allocation."""
    rows = []
    with (ROOT / "results" / "test" / "metrics_by_mcs_rb.csv").open() as handle:
        for row in csv.DictReader(handle):
            rows.append({k: float(v) for k, v in row.items()})
    target_rb = 52
    rows = [r for r in rows if int(r["num_rbs"]) == target_rb]
    labels = {
        6: "MCS 6 / QPSK (Qm=2)",
        13: "MCS 13 / 16QAM (Qm=4)",
        24: "MCS 24 / 64QAM (Qm=6)",
    }
    colors = {6: "#1f77b4", 13: "#ff7f0e", 24: "#2ca02c"}
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for mcs in sorted({int(r["mcs_index"]) for r in rows}):
        subset = [r for r in rows if int(r["mcs_index"]) == mcs]
        subset.sort(key=lambda r: r["snr_db"])
        ax.semilogy(
            [r["snr_db"] for r in subset],
            [r["ber"] for r in subset],
            marker="o",
            ms=4,
            color=colors.get(mcs),
            label=labels.get(mcs, f"MCS {mcs}"),
        )
    ax.set(
        xlabel="SNR (dB)",
        ylabel="Coded BER",
        title=f"调制方式自适应 coded BER vs SNR (RB={target_rb})",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_modulation_ber_snr.png")
    plt.close(fig)
    print("wrote fig_modulation_ber_snr.png")


def fig_evm_snr() -> None:
    """EVM vs SNR: neural vs traditional, averaged over RB."""
    rows = []
    with (ROOT / "results" / "test" / "metrics_by_mcs_rb.csv").open() as handle:
        for row in csv.DictReader(handle):
            rows.append({k: float(v) for k, v in row.items()})
    snrs = sorted({r["snr_db"] for r in rows})
    neural = {s: np.mean([r["neural_evm"] for r in rows if r["snr_db"] == s]) for s in snrs}
    traditional = {
        s: np.mean([r["traditional_evm"] for r in rows if r["snr_db"] == s]) for s in snrs
    }
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.semilogy(
        snrs,
        [100 * traditional[s] for s in snrs],
        marker="s",
        ms=4,
        label="传统均衡器",
    )
    ax.semilogy(snrs, [100 * neural[s] for s in snrs], marker="o", ms=4, label="神经接收机")
    ax.set(
        xlabel="SNR (dB)",
        ylabel="EVM (%)",
        title="测试集 EVM vs SNR(各 RB 平均,对数坐标)",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_evm_snr.png")
    plt.close(fig)
    print("wrote fig_evm_snr.png")


def fig_constellation() -> None:
    """Constellation comparison from results/constellation.npz."""
    from matplotlib.ticker import MaxNLocator

    data = np.load(ROOT / "results" / "constellation.npz")
    mcs = int(data["mcs_index"][0])
    num_rbs = int(data["num_rbs"][0])
    rng = np.random.default_rng(42)
    # Sample a fixed number of points per panel for readability.
    limit = 8000
    pick = rng.choice(len(data["reference"]), size=min(limit, len(data["reference"])), replace=False)
    ref = data["reference"][pick]
    neu = data["neural"][pick]
    trad = data["traditional"][pick]

    fig, axes = plt.subplots(
        1, 3, figsize=(12.6, 4.4), constrained_layout=True
    )
    panels = [
        (axes[0], ref, "参考发送符号"),
        (axes[1], neu, f"神经接收机输出\n(EVM {data['neural_evm_percent'][0]:.1f}%)"),
        (axes[2], trad, f"传统均衡器输出\n(EVM {data['traditional_evm_percent'][0]:.1f}%)"),
    ]
    bound = 1.55  # 16QAM 归一化星座点幅度为 1,留出余量并保证方形
    for ax, points, title in panels:
        ax.scatter(points.real, points.imag, s=1.5, alpha=0.5, c="#1f77b4", linewidths=0)
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_box_aspect(1)  # 保证坐标格为正方形
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=7.5, pad=2)
        ax.set(xlabel="I", ylabel="Q", title=title)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
    fig.suptitle(f"星座图对比:MCS {mcs}(QPSK), {num_rbs} RB", fontsize=12)
    fig.savefig(OUT / "fig_constellation.png")
    plt.close(fig)
    print("wrote fig_constellation.png")


def fig_pipeline_demo() -> None:
    """Schematic gain bars: EVM improvement vs SNR from test metrics."""
    rows = []
    with (ROOT / "results" / "test" / "metrics_by_mcs_rb.csv").open() as handle:
        for row in csv.DictReader(handle):
            rows.append({k: float(v) for k, v in row.items()})
    snrs = sorted({r["snr_db"] for r in rows})
    trad = {s: np.mean([r["traditional_evm"] for r in rows if r["snr_db"] == s]) for s in snrs}
    neu = {s: np.mean([r["neural_evm"] for r in rows if r["snr_db"] == s]) for s in snrs}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(snrs))
    width = 0.36
    ax.bar(x - width / 2, [100 * trad[s] for s in snrs], width, label="传统均衡器 EVM")
    ax.bar(x + width / 2, [100 * neu[s] for s in snrs], width, label="神经接收机 EVM")
    ax.set_xticks(x, [f"{s:.0f}" for s in snrs])
    ax.set(xlabel="SNR (dB)", ylabel="EVM (%)", title="EVM 对比(按 SNR,各 RB 平均)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_evm_bar.png")
    plt.close(fig)
    print("wrote fig_evm_bar.png")


def _load_test_metrics() -> list[dict[str, float]]:
    rows = []
    with (ROOT / "results" / "test" / "metrics_by_mcs_rb.csv").open() as handle:
        for row in csv.DictReader(handle):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def fig_rb_adaptation() -> None:
    """RB adaptation result: performance across allocation widths."""
    rows = _load_test_metrics()
    rbs = sorted({int(r["num_rbs"]) for r in rows})
    snrs = sorted({r["snr_db"] for r in rows})
    # Use the middle SNR as a direct width comparison and show all-SNR spread.
    mid_snr = snrs[len(snrs) // 2]
    selected = [r for r in rows if r["snr_db"] == mid_snr]
    selected.sort(key=lambda r: (r["mcs_index"], r["num_rbs"]))
    mcs_values = sorted({int(r["mcs_index"]) for r in selected})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    ax = axes[0]
    for mcs in mcs_values:
        subset = [r for r in selected if int(r["mcs_index"]) == mcs]
        subset.sort(key=lambda r: r["num_rbs"])
        ax.plot(
            [r["num_rbs"] for r in subset],
            [100 * r["neural_evm"] for r in subset],
            marker="o",
            label=f"MCS {mcs}",
        )
    ax.set(xlabel="RB allocation", ylabel="Neural EVM (%)", title=f"(a) 固定 SNR={mid_snr:.0f} dB")
    ax.set_xticks(rbs, [str(rb) for rb in rbs])
    ax.grid(alpha=0.3)
    ax.legend(title="调制场景")

    ax = axes[1]
    # Relative gain is computed against the supplied traditional_eq baseline.
    gains = []
    labels = []
    for rb in rbs:
        subset = [r for r in rows if int(r["num_rbs"]) == rb]
        gains.append(100 * (1 - np.mean([r["neural_evm"] for r in subset]) /
                            np.mean([r["traditional_evm"] for r in subset])))
        labels.append(str(rb))
    ax.bar(labels, gains, color="#2ca02c")
    ax.set(xlabel="RB allocation", ylabel="EVM improvement (%)", title="(b) 跨 SNR、MCS 的平均改善")
    ax.axhline(0, color="k", lw=0.7)
    ax.grid(alpha=0.3, axis="y")
    for index, value in enumerate(gains):
        ax.text(index, value + 0.8, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.suptitle("RB 自适应结果：共享全卷积权重覆盖不同带宽", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rb_adaptation.png")
    plt.close(fig)
    print("wrote fig_rb_adaptation.png")


def fig_modulation_adaptation() -> None:
    """Modulation adaptation result: MCS/Qm curves at a common RB width."""
    rows = _load_test_metrics()
    mcs_values = sorted({int(r["mcs_index"]) for r in rows})
    rb = 52 if any(int(r["num_rbs"]) == 52 for r in rows) else sorted({int(r["num_rbs"]) for r in rows})[0]
    snrs = sorted({r["snr_db"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    labels = {6: "MCS 6 / QPSK (Qm=2)", 13: "MCS 13 / 16QAM (Qm=4)", 24: "MCS 24 / 64QAM (Qm=6)"}
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for mcs, color in zip(mcs_values, colors):
        subset = [r for r in rows if int(r["mcs_index"]) == mcs and int(r["num_rbs"]) == rb]
        subset.sort(key=lambda r: r["snr_db"])
        label = labels.get(mcs, f"MCS {mcs}")
        axes[0].semilogy([r["snr_db"] for r in subset], [r["ber"] for r in subset], marker="o", color=color, label=label)
        axes[1].plot([r["snr_db"] for r in subset], [100 * r["neural_evm"] for r in subset], marker="o", color=color, label=label)
    axes[0].set(xlabel="SNR (dB)", ylabel="Coded BER", title=f"(a) RB={rb} 的软比特输出")
    axes[1].set(xlabel="SNR (dB)", ylabel="Neural EVM (%)", title=f"(b) RB={rb} 的星座均衡输出")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("调制方式自适应结果：同一网络权重覆盖 Qm=2/4/6", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_modulation_adaptation.png")
    plt.close(fig)
    print("wrote fig_modulation_adaptation.png")


def fig_other_modulation_constellations() -> None:
    """Reference/traditional constellations for the other stored modulations."""
    import h5py

    scenarios = [(13, "16QAM"), (24, "64QAM")]
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.0), constrained_layout=True)
    for row, (mcs, modulation) in enumerate(scenarios):
        paths = sorted((ROOT / "dataset_h5").glob(f"*mcs{mcs:03d}*.h5"))
        if not paths:
            raise FileNotFoundError(f"No HDF5 shard found for MCS {mcs}")
        references = []
        traditionals = []
        for path in paths[: min(3, len(paths))]:
            with h5py.File(path, "r") as handle:
                axis_order = handle.attrs.get("axis_order", "matlab")
                if isinstance(axis_order, bytes):
                    axis_order = axis_order.decode()
                count = min(5, handle["tx_symbols"].shape[-1] if axis_order == "matlab" else handle["tx_symbols"].shape[0])
                for index in range(count):
                    def read(name: str) -> np.ndarray:
                        value = np.asarray(handle[name][..., index] if axis_order == "matlab" else handle[name][index])
                        if axis_order == "matlab":
                            value = value.transpose(tuple(reversed(range(value.ndim))))
                        return value
                    tx = read("tx_symbols")
                    eq = read("traditional_eq")
                    mask = read("data_mask").astype(bool)
                    references.append(tx[..., 0, 0] + 1j * tx[..., 0, 1])
                    traditionals.append(eq[..., 0, 0] + 1j * eq[..., 0, 1])
                    # Keep only data REs from layer 1; this avoids duplicating points.
                    references[-1] = references[-1][mask[..., 0]]
                    traditionals[-1] = traditionals[-1][mask[..., 0]]
        ref = np.concatenate(references)
        trad = np.concatenate(traditionals)
        pick = rng.choice(len(ref), size=min(7000, len(ref)), replace=False)
        ref, trad = ref[pick], trad[pick]
        bound = 1.55 if mcs == 13 else 1.25
        for col, (ax, points, title) in enumerate(
            ((axes[row, 0], ref, f"MCS {mcs} / {modulation}\n参考发送符号"),
             (axes[row, 1], trad, f"MCS {mcs} / {modulation}\n传统均衡器输出"))
        ):
            ax.scatter(points.real, points.imag, s=1.3, alpha=0.45, c="#1f77b4", linewidths=0)
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-bound, bound)
            ax.set_box_aspect(1)
            ax.set(xlabel="I", ylabel="Q", title=title)
            ax.grid(alpha=0.3)
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5)
    fig.suptitle("其他调制方式星座图：16QAM 与 64QAM", fontsize=12)
    fig.savefig(OUT / "fig_other_modulation_constellations.png", dpi=180)
    plt.close(fig)
    print("wrote fig_other_modulation_constellations.png")


if __name__ == "__main__":
    fig_training()
    fig_ber_snr()
    fig_modulation_ber_snr()
    fig_evm_snr()
    fig_constellation()
    fig_pipeline_demo()
    fig_rb_adaptation()
    fig_modulation_adaptation()
    fig_other_modulation_constellations()
    print("all figures done")
