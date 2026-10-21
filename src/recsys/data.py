"""Synthetic implicit-feedback data with known structure, plus leakage-free splitting.

Items belong to latent categories and users prefer a few categories, on top of a global
popularity long-tail. Because the ground truth is known, we can sanity-check that the
model recovers it (e.g. similar items share a category).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import DataConfig


def generate_data(cfg: DataConfig, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (interactions, items). Interactions: user_id, item_id, timestamp."""
    rng = np.random.default_rng(seed)
    cat_of_item = rng.integers(0, cfg.n_categories, cfg.n_items)
    popularity = rng.lognormal(mean=0.0, sigma=1.0, size=cfg.n_items)

    # Each user likes 1-3 categories.
    prefs = np.full((cfg.n_users, cfg.n_categories), 0.05)
    for u in range(cfg.n_users):
        liked = rng.choice(cfg.n_categories, size=rng.integers(1, 4), replace=False)
        prefs[u, liked] = rng.uniform(0.5, 1.0, size=len(liked))
    prefs /= prefs.sum(axis=1, keepdims=True)

    rows = []
    for u in range(cfg.n_users):
        p = popularity * prefs[u, cat_of_item]
        p /= p.sum()
        n = min(cfg.n_items - 1, 5 + rng.poisson(cfg.mean_interactions - 5))
        items = rng.choice(cfg.n_items, size=n, replace=False, p=p)
        ts = np.sort(rng.integers(1_600_000_000, 1_700_000_000, size=n))
        rows.append(pd.DataFrame({"user_id": u, "item_id": items, "timestamp": ts}))
    interactions = pd.concat(rows, ignore_index=True)

    items_df = pd.DataFrame(
        {
            "item_id": np.arange(cfg.n_items),
            "category": [f"cat_{c}" for c in cat_of_item],
        }
    )
    return interactions, items_df


@dataclass
class Split:
    train: sp.csr_matrix  # all interactions before validation item
    val_target: np.ndarray  # item index per user, -1 where user has no validation item
    test_target: np.ndarray
    n_users: int
    n_items: int


def to_matrix(df: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    m = sp.csr_matrix(
        (np.ones(len(df), dtype=np.float32), (df["user_id"].to_numpy(), df["item_id"].to_numpy())),
        shape=(n_users, n_items),
    )
    m.data[:] = 1.0  # binary, duplicates collapsed
    return m


def leave_last_out(df: pd.DataFrame, n_users: int, n_items: int) -> Split:
    """Per user: last interaction -> test, second-last -> validation, rest -> train.
    Users with fewer than 3 interactions are train-only (never evaluated)."""
    df = df.sort_values(["user_id", "timestamp", "item_id"], kind="stable")
    rank_from_end = df.groupby("user_id").cumcount(ascending=False)
    size = df.groupby("user_id")["item_id"].transform("size")
    eligible = size >= 3
    is_test = (rank_from_end == 0) & eligible
    is_val = (rank_from_end == 1) & eligible

    val_target = np.full(n_users, -1, dtype=np.int64)
    test_target = np.full(n_users, -1, dtype=np.int64)
    test_target[df.loc[is_test, "user_id"].to_numpy()] = df.loc[is_test, "item_id"].to_numpy()
    val_target[df.loc[is_val, "user_id"].to_numpy()] = df.loc[is_val, "item_id"].to_numpy()
    train = to_matrix(df[~(is_test | is_val)], n_users, n_items)
    return Split(train, val_target, test_target, n_users, n_items)


def add_targets(train: sp.csr_matrix, targets: np.ndarray) -> sp.csr_matrix:
    """Return train plus the given held-out items (used to mask seen items at test time)."""
    users = np.flatnonzero(targets >= 0)
    extra = sp.csr_matrix(
        (np.ones(len(users), dtype=np.float32), (users, targets[users])), shape=train.shape
    )
    out = (train + extra).tocsr()
    out.data[:] = 1.0
    return out
