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
        title="测试集 coded BER vs SNR(MCS 6 / 16QAM)",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="RB 分配")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ber_snr.png")
    plt.close(fig)
    print("wrote fig_ber_snr.png")


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
    fig.suptitle(f"星座图对比:MCS {mcs}(16QAM), {num_rbs} RB", fontsize=12)
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


if __name__ == "__main__":
    fig_training()
    fig_ber_snr()
    fig_evm_snr()
    fig_constellation()
    fig_pipeline_demo()
    print("all figures done")
