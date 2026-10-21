import numpy as np
import pytest
import scipy.sparse as sp

from recsys.models import ALS, ItemKNN, Popularity, build_model


def toy():
    # Two clear taste groups: items {0,1,2} vs {3,4,5}
    rows = [[1, 1, 1, 0, 0, 0]] * 20 + [[0, 0, 0, 1, 1, 1]] * 20
    return sp.csr_matrix(np.array(rows, dtype=np.float32))


def test_popularity_scores_follow_counts():
    R = sp.csr_matrix(np.array([[1, 1, 0], [1, 0, 0]], dtype=np.float32))
    s = Popularity().fit(R).score(np.array([0]))
    assert list(s[0]) == [2.0, 1.0, 0.0]


@pytest.mark.parametrize("model", [ItemKNN(neighbors=3), ALS(factors=3, reg=1.0, alpha=5, iterations=10)])
def test_models_learn_taste_groups(model):
    R = toy()
    model.fit(R)
    # a user who has only seen item 0 should prefer items 1,2 over 3,4,5
    partial = sp.csr_matrix(np.array([[1, 0, 0, 0, 0, 0]], dtype=np.float32))
    if isinstance(model, ItemKNN):
        model.R = partial
        s = model.score(np.array([0]))[0]
    else:
        s = model.score(np.array([0]))[0]  # user 0 is in group A
    assert min(s[1], s[2]) > max(s[3], s[4], s[5])


def test_als_is_deterministic():
    a = ALS(factors=3, iterations=5, seed=1).fit(toy()).V
    b = ALS(factors=3, iterations=5, seed=1).fit(toy()).V
    assert np.allclose(a, b)


def test_similar_items_in_same_group():
    top, _ = ALS(factors=3, iterations=10).fit(toy()).similar_items(0, k=2)
    assert set(top) == {1, 2}


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("nope")
    with pytest.raises(NotImplementedError):
        Popularity().fit(toy()).similar_items(0)
