import numpy as np

from recsys.config import DataConfig
from recsys.data import add_targets, generate_data, leave_last_out

CFG = DataConfig(n_users=100, n_items=60, n_categories=3, mean_interactions=15)


def test_generation_is_deterministic_and_valid():
    a, items = generate_data(CFG, seed=3)
    b, _ = generate_data(CFG, seed=3)
    assert a.equals(b)
    assert a["item_id"].between(0, 59).all()
    assert not a.duplicated(["user_id", "item_id"]).any()
    assert a.groupby("user_id").size().min() >= 5
    assert items["category"].nunique() <= 3


def test_split_has_no_leakage_and_is_chronological():
    df, _ = generate_data(CFG, seed=4)
    s = leave_last_out(df, 100, 60)
    users = np.flatnonzero(s.test_target >= 0)
    assert len(users) == 100
    for u in users[:25]:
        seen = set(s.train[u].indices)
        assert s.test_target[u] not in seen
        assert s.val_target[u] not in seen
        assert s.test_target[u] != s.val_target[u]
        hist = df[df.user_id == u].sort_values(["timestamp", "item_id"], kind="stable")
        assert hist["item_id"].iloc[-1] == s.test_target[u]  # last item is the test item
        assert hist["item_id"].iloc[-2] == s.val_target[u]


def test_add_targets_restores_items():
    df, _ = generate_data(CFG, seed=5)
    s = leave_last_out(df, 100, 60)
    full = add_targets(s.train, s.val_target)
    assert full.nnz == s.train.nnz + (s.val_target >= 0).sum()
