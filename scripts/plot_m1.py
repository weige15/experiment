"""Reproducible whole-set heatmaps and accuracy CIs from measured CSVs only."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
import pandas as pd

from m1_data import SOURCES, save_json
from m1_run import file_hash, read_json

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "pdf.fonttype": 42,
    "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})


def heatmap(frame, column, title, path, normalized=False):
    order = {d: i for i, d in enumerate(SOURCES)}
    ids = sorted(frame.request_id.unique(), key=lambda r: (order[r.split(":")[0]], int(r.split(":")[-1])))
    matrices = [frame[frame.wording == w].pivot(index="request_id", columns="group", values=column)
                .reindex(index=ids, columns=range(8)).to_numpy() for w in ["a", "b"]]
    if normalized:
        matrices = [m / np.nanmean(m, axis=1, keepdims=True) for m in matrices]
        norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=max(2., np.nanmax(matrices)))
        cmap, label = "RdBu_r", "KL / request's across-group mean (1 = mean)"
    else:
        # Explicit plotting floor only; raw data and all statistics remain unclipped.
        minimum = max(1e-8, np.nanmin(matrices))
        norm = LogNorm(vmin=minimum, vmax=max(minimum * 1.01, np.nanmax(matrices)))
        cmap, label = "viridis", "KL(BF16 || W4), nats/token; log color scale"
    fig, axes = plt.subplots(1, 2, figsize=(10, max(5.5, len(ids) * .17)), layout="constrained", sharey=True)
    for ax, matrix, wording in zip(axes, matrices, ["A", "B"]):
        image = ax.imshow(np.maximum(matrix, 1e-8) if not normalized else matrix,
                          aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_title(f"Wording {wording}")
        ax.set_xticks(range(8), [f"g{g}\n{g*4}–{g*4+3}" for g in range(8)], fontsize=8)
        ax.set_xlabel("W4 group / zero-indexed layer range; all other groups BF16")
        ax.set_yticks(range(len(ids)), ids, fontsize=6)
        for i in range(1, len(ids)):
            if ids[i].split(":")[0] != ids[i-1].split(":")[0]:
                ax.axhline(i - .5, color="black", lw=1.2)
    axes[0].set_ylabel("Every source request (dataset/index order, not effect-size order)")
    fig.colorbar(image, ax=axes, shrink=.75, label=label, pad=.02)
    fig.suptitle(title, fontsize=12)
    for extension in ["png", "pdf"]:
        fig.savefig(path.with_suffix("." + extension), bbox_inches="tight")
    plt.close(fig)


def accuracy_plot(stats, meta, path):
    table = stats[(stats.metric == "full") & (stats.scope == "all") & (stats.wording == "paired")].set_index("statistic")
    labels = [
        ("BF16_accuracy", "BF16 reference (not budget-matched)", "#666666"),
        ("calibration_KL_fixed_accuracy", f"Calibration KL fixed (g{meta['fixed_groups']['KL']})", "#0072B2"),
        ("calibration_accuracy_fixed_accuracy", f"Calibration accuracy fixed (g{meta['fixed_groups']['accuracy']})", "#0072B2"),
        ("evaluation_KL_best_fixed_ORACLE_accuracy", "Evaluation KL-best fixed [oracle]", "#E69F00"),
        ("evaluation_accuracy_best_fixed_ORACLE_accuracy", "Evaluation accuracy-best fixed [oracle]", "#E69F00"),
        ("per_request_KL_ORACLE_accuracy", "Per-request KL [oracle]", "#CC79A7"),
        ("per_request_label_ORACLE_accuracy", "Per-request correctness [label oracle]", "#CC79A7"),
        ("cross_wording_KL_ORACLE_diagnostic_accuracy", "Other-wording KL [oracle diagnostic]", "#009E73"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for i, (key, label, color) in enumerate(labels):
        r = table.loc[key]
        ax.hlines(i, r.ci_low * 100, r.ci_high * 100, color=color, lw=2)
        ax.plot(r.estimate * 100, i, "o", color=color, ms=6)
    ax.set_yticks(range(len(labels)), [r[1] for r in labels])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Task correctness (%); 95% source-cluster bootstrap CI")
    ax.grid(axis="x", alpha=.2)
    ax.set_title(f"{meta['role'].upper()}: all {meta['source_requests']} sources, paired wordings; no deployable router")
    for extension in ["png", "pdf"]:
        fig.savefig(path.with_suffix("." + extension), bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", default="results/m1/analysis")
    p.add_argument("--out", default="results/m1")
    args = p.parse_args()
    source, out = Path(args.analysis), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = read_json(source / "analysis_metadata.json")
    frame = pd.read_csv(source / "per_request_measurements.csv")
    stats = pd.read_csv(source / "statistics.csv")
    title = f"{meta['role'].upper()} · all {meta['source_requests']} source requests · W4 layer-group sensitivity"
    heatmap(frame, "kl_mean", title + "\nFull BF16 continuation (primary)", out / "sensitivity_heatmap")
    heatmap(frame, "kl_content_mean", title + "\nNon-whitespace content only; no next-question delimiter/EOS", out / "sensitivity_heatmap_content")
    heatmap(frame, "kl_mean", title + "\nRow-normalized full-continuation profile", out / "sensitivity_heatmap_normalized", normalized=True)
    accuracy_plot(stats, meta, out / "accuracy_comparison")
    save_json(out / "figure_metadata.json", dict(role=meta["role"], source_requests=meta["source_requests"],
        raw_color_floor=1e-8, row_order="dataset then original source index; never sorted by outcome",
        inputs={str(source / name): file_hash(source / name) for name in ["statistics.csv", "per_request_measurements.csv"]},
        code_sha256=file_hash(__file__)))
    print(f"Saved 4 PNG/PDF figure pairs from real {meta['role']} measurements to {out}")


if __name__ == "__main__":
    main()
