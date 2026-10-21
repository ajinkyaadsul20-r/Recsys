"""Ranking metrics for leave-one-out evaluation, with seen-item masking."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def target_ranks(scores: np.ndarray, seen: sp.csr_matrix, targets: np.ndarray) -> np.ndarray:
    """0-based rank of each user's held-out item among all unseen items (ties averaged).
    Only users with targets >= 0 are returned."""
    users = np.flatnonzero(targets >= 0)
    S = scores[users].astype(np.float64).copy()
    rows, cols = seen[users].nonzero()
    S[rows, cols] = -np.inf
    tgt = S[np.arange(len(users)), targets[users]][:, None]
    greater = (S > tgt).sum(axis=1)
    equal = (S == tgt).sum(axis=1) - 1  # exclude the target itself
    return greater + equal / 2.0


def metrics_from_ranks(ranks: np.ndarray, k: int) -> dict[str, float]:
    hit = ranks < k
    return {
        f"hit_rate@{k}": float(hit.mean()),  # == recall@k with one held-out item
        f"ndcg@{k}": float(np.where(hit, 1.0 / np.log2(ranks + 2.0), 0.0).mean()),
        "mrr": float((1.0 / (ranks + 1.0)).mean()),
    }


def hit_rate_curve(ranks: np.ndarray, max_k: int = 50) -> list[float]:
    return [float((ranks < k).mean()) for k in range(1, max_k + 1)]


def coverage_at_k(scores: np.ndarray, seen: sp.csr_matrix, k: int, users: np.ndarray) -> float:
    """Share of the catalogue that appears in at least one user's top-k list."""
    S = scores[users].astype(np.float64).copy()
    rows, cols = seen[users].nonzero()
    S[rows, cols] = -np.inf
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    return float(len(np.unique(top)) / scores.shape[1])


def evaluate(
    model, train: sp.csr_matrix, seen: sp.csr_matrix, targets: np.ndarray, k: int
) -> tuple[dict[str, float], np.ndarray]:
    users = np.arange(train.shape[0])
    scores = model.score(users)
    ranks = target_ranks(scores, seen, targets)
    out = metrics_from_ranks(ranks, k)
    out[f"coverage@{k}"] = coverage_at_k(scores, seen, k, np.flatnonzero(targets >= 0))
    return out, ranks
