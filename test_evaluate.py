import numpy as np
import scipy.sparse as sp

from recsys.evaluate import metrics_from_ranks, target_ranks


def test_perfect_and_worst_ranking():
    scores = np.array([[0.1, 0.9, 0.5], [0.9, 0.1, 0.5]])
    seen = sp.csr_matrix((2, 3))
    ranks = target_ranks(scores, seen, np.array([1, 1]))
    assert list(ranks) == [0, 2]
    m = metrics_from_ranks(ranks, k=1)
    assert m["hit_rate@1"] == 0.5
    assert abs(m["mrr"] - (1 + 1 / 3) / 2) < 1e-9


def test_seen_items_are_masked():
    scores = np.array([[0.9, 0.5, 0.1]])
    seen = sp.csr_matrix(np.array([[1, 0, 0]]))  # best-scored item already seen
    assert target_ranks(scores, seen, np.array([1]))[0] == 0


def test_ties_are_averaged_and_users_without_target_skipped():
    scores = np.ones((2, 4))
    ranks = target_ranks(scores, sp.csr_matrix((2, 4)), np.array([2, -1]))
    assert list(ranks) == [1.5]


def test_ndcg_formula():
    m = metrics_from_ranks(np.array([0.0, 1.0]), k=10)
    assert abs(m["ndcg@10"] - (1.0 + 1 / np.log2(3)) / 2) < 1e-9
