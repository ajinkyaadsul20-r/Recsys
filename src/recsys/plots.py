"""Result plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"popularity": "#9aa5b1", "item_knn": "#e08a3c", "als": "#3b6fb6"}


def save_plots(results: list[dict[str, Any]], items: pd.DataFrame, model, k: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for r in results:
        ax.plot(range(1, len(r["hit_curve"]) + 1), r["hit_curve"], label=r["model"],
                color=COLORS.get(r["model"]), lw=2)
    ax.set_xlabel("k (list length)")
    ax.set_ylabel("Hit rate@k (test)")
    ax.set_title("Held-out item found in top-k")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "hit_rate_curve.png", dpi=150)
    plt.close(fig)

    metric = f"ndcg@{k}"
    names = [r["model"] for r in results]
    vals = [r["test"][metric] for r in results]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.bar(names, vals, color=[COLORS.get(n, "#555") for n in names])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylabel(f"{metric} (test)")
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)

    if hasattr(model, "V"):  # 2-D PCA of item factors coloured by (hidden) category
        V = model.V - model.V.mean(axis=0)
        _, _, vt = np.linalg.svd(V, full_matrices=False)
        xy = V @ vt[:2].T
        cats = items["category"].astype("category")
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=cats.cat.codes, cmap="tab10", s=14)
        ax.set_title("Learned item embeddings (PCA), coloured by true category")
        ax.legend(*sc.legend_elements(), title="category", fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(out / "item_embeddings.png", dpi=150)
        plt.close(fig)
