from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .utils import atomic_json_dump, atomic_write_text, read_tsv


def _load_history(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key == "epoch":
                row[key] = int(value)
            elif value == "":
                row[key] = float("nan")
            else:
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
    return rows


def _series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)


def generate_report(run_dir: str | Path) -> list[Path]:
    run = Path(run_dir)
    figures = run / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = figures / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("生成静态图需要 matplotlib；请在 rec_rank 环境安装 requirements.txt") from exc

    history = _load_history(run / "logs" / "history.csv")
    if not history:
        raise ValueError("history.csv 为空")
    epochs = _series(history, "epoch")
    best = json.loads((run / "evaluation" / "best_metrics.json").read_text(encoding="utf-8"))
    best_epoch = int(best["epoch"])
    generated: list[Path] = []

    def save(fig: Any, name: str) -> None:
        path = figures / name
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        generated.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, _series(history, "train_loss"), label="train BCE")
    ax.plot(epochs, _series(history, "test_loss"), label="test BCE")
    ax.axvline(best_epoch, color="black", linestyle="--", alpha=0.5, label=f"best={best_epoch}")
    ax.set(xlabel="Epoch", ylabel="BCE loss", title="Training and test labeled-pair loss")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "loss_curves.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    recall_values = []
    for k in (1, 5, 10):
        values = _series(history, f"recall_at_{k}")
        recall_values.extend(values[np.isfinite(values)].tolist())
        ax.plot(epochs, values, label=f"Recall@{k}")
    ax.axvline(best_epoch, color="black", linestyle="--", alpha=0.5)
    recall_upper = max(0.05, min(1.0, max(recall_values, default=0.05) * 1.2))
    ax.set(
        ylabel="Recall",
        ylim=(0, recall_upper),
        title="Exact-protein retrieval recall",
    )
    ax.set_xlabel("Epoch", loc="center")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "recall_by_epoch.png")

    fig, left = plt.subplots(figsize=(8, 5))
    right = left.twinx()
    left.plot(epochs, _series(history, "mrr"), color="tab:blue", label="MRR")
    right.plot(epochs, _series(history, "median_rank"), color="tab:orange", label="Median rank")
    left.axvline(best_epoch, color="black", linestyle="--", alpha=0.5)
    left.set(xlabel="Epoch", ylabel="MRR", title="Retrieval rank metrics")
    right.set_ylabel("Median dense rank")
    left.grid(alpha=0.25)
    left.legend(loc="upper left")
    right.legend(loc="upper right")
    save(fig, "rank_metrics.png")

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(epochs, _series(history, "gamma"), label="gamma")
    axes[0].plot(epochs, _series(history, "bias"), label="bias")
    axes[0].axvline(best_epoch, color="black", linestyle="--", alpha=0.5)
    axes[0].set(ylabel="Value", title="BCE logit calibration parameters")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    norm_mean_line = axes[1].plot(
        epochs, _series(history, "gradient_norm_mean"), label="pre-clip mean"
    )[0]
    norm_max_line = axes[1].plot(
        epochs, _series(history, "gradient_norm_max"), label="pre-clip max"
    )[0]
    clip_axis = axes[1].twinx()
    clip_values = _series(history, "gradient_clip_fraction")
    clip_line = clip_axis.plot(
        epochs,
        clip_values,
        color="tab:green",
        linestyle="--",
        label="clip fraction",
    )[0]
    finite_clip = clip_values[np.isfinite(clip_values)]
    clip_upper = max(0.05, float(finite_clip.max()) * 1.2 if finite_clip.size else 0.05)
    clip_axis.set_ylim(0, clip_upper)
    clip_axis.set_ylabel("Clipped-step fraction")
    axes[1].set(
        xlabel="Epoch",
        ylabel="Pre-clip gradient norm",
        title="Gradient clipping diagnostics",
    )
    axes[1].legend(
        [norm_mean_line, norm_max_line, clip_line],
        ["pre-clip mean", "pre-clip max", "clip fraction"],
    )
    axes[1].grid(alpha=0.25)
    save(fig, "gamma_and_gradients.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, _series(history, "train_positive_cosine_mean"), label="positive cosine")
    ax.plot(epochs, _series(history, "train_negative_cosine_mean"), label="negative cosine")
    ax.plot(epochs, _series(history, "train_pair_margin_mean"), label="within-protein margin")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(best_epoch, color="black", linestyle="--", alpha=0.5)
    ax.set(xlabel="Epoch", ylabel="Cosine / margin", title="Training compatibility separation")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "cosine_and_margin.png")

    last_epoch = int(max(epochs))
    middle_epoch = last_epoch // 2
    distribution_epochs = list(dict.fromkeys([0, middle_epoch, best_epoch]))
    fig, axes = plt.subplots(1, len(distribution_epochs), figsize=(5 * len(distribution_epochs), 4), sharex=True, sharey=True)
    if len(distribution_epochs) == 1:
        axes = [axes]
    for ax, epoch in zip(axes, distribution_epochs, strict=True):
        payload = torch.load(
            run / "evaluation" / f"labeled_cosines_epoch_{epoch:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        ax.hist(payload["positive"].numpy(), bins=50, alpha=0.55, density=True, label="positive")
        ax.hist(payload["negative"].numpy(), bins=50, alpha=0.55, density=True, label="negative")
        ax.set(title=f"Epoch {epoch}", xlabel="Cosine", ylabel="Density")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Test labeled-pair cosine distributions")
    save(fig, "cosine_distributions.png")

    query_rows = list(read_tsv(run / "evaluation" / "best_query_metrics.tsv"))
    best_ranks = np.asarray([int(row["best_positive_rank"]) for row in query_rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(best_ranks, bins=min(80, max(10, int(np.sqrt(len(best_ranks))))))
    axes[0].set(xlabel="Best known-positive dense rank", ylabel="Queries", title="Rank histogram")
    sorted_ranks = np.sort(best_ranks)
    axes[1].step(sorted_ranks, np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks), where="post")
    axes[1].set(xlabel="Best known-positive dense rank", ylabel="Cumulative fraction", title="Rank CDF")
    for ax in axes:
        ax.grid(alpha=0.25)
    save(fig, "best_rank_distribution.png")

    recall_rows = list(read_tsv(run / "evaluation" / f"recall_curve_epoch_{best_epoch:03d}.tsv"))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([int(row["k"]) for row in recall_rows], [float(row["recall"]) for row in recall_rows])
    ax.axvline(10, color="black", linestyle="--", alpha=0.5, label="TOP10")
    ax.set(xlabel="K (dense rank threshold)", ylabel="Recall@K", ylim=(0, 1), title=f"Recall@K at best epoch {best_epoch}")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, "recall_at_k.png")

    report_lines = [
        f"# RecRank V1 run report: {run.name}",
        "",
        f"- Best epoch: {best_epoch}",
        f"- Recall@10: {best['metrics']['recall_at_10']:.6f}",
        f"- MRR: {best['metrics']['mrr']:.6f}",
        f"- Median dense rank: {best['metrics']['median_rank']:.3f}",
        "- 注意：本轮实验按既定方案使用测试集逐 epoch 选 best，因此这些数值不是未触碰测试集上的无偏最终估计。",
        "",
        "## Figures",
        "",
    ]
    report_lines.extend(f"- [{path.name}]({path.name})" for path in generated)
    atomic_write_text(figures / "REPORT.md", "\n".join(report_lines) + "\n")
    generated.append(figures / "REPORT.md")
    atomic_json_dump(figures / "manifest.json", {"best_epoch": best_epoch, "files": [path.name for path in generated]})
    generated.append(figures / "manifest.json")
    return generated
