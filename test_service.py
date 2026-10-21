import json

import pytest

from recsys.service import RecommenderService


def test_artifacts_written(trained_dir):
    assert (trained_dir / "bundle.joblib").exists()
    for p in ("hit_rate_curve", "model_comparison"):
        assert (trained_dir / "plots" / f"{p}.png").exists()
    meta = json.loads((trained_dir / "metadata.json").read_text())
    board = {r["model"]: r["test"]["ndcg@10"] for r in meta["leaderboard"]}
    assert board["als"] > board["popularity"]
    assert board["item_knn"] > board["popularity"]


def test_recommend_excludes_seen_items(trained_dir):
    svc = RecommenderService(trained_dir)
    for user in (0, 7, 42):
        out = svc.recommend(user, k=10)
        ids = [i["item_id"] for i in out["items"]]
        assert len(ids) == len(set(ids)) == 10
        assert not set(ids) & set(svc.R[user].indices)
        assert out["cold_start"] is False


def test_cold_start_users_get_popular_items(trained_dir):
    svc = RecommenderService(trained_dir)
    out = svc.recommend(10_000, k=5)
    assert out["cold_start"] and out["strategy"] == "popularity"
    assert len(out["items"]) == 5


def test_invalid_k(trained_dir):
    with pytest.raises(ValueError):
        RecommenderService(trained_dir).recommend(0, k=0)


def test_similar_items_share_category(trained_dir):
    svc = RecommenderService(trained_dir)
    cats = svc.items["category"].to_numpy()
    matches = total = 0
    for item in range(0, svc.n_items, 6):
        for s in svc.similar(item, k=5)["similar"]:
            matches += cats[s["item_id"]] == cats[item]
            total += 1
    assert matches / total > 0.5  # chance level is ~0.25 with 4 categories


def test_unknown_item(trained_dir):
    with pytest.raises(KeyError):
        RecommenderService(trained_dir).similar(10_000)


def test_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        RecommenderService(tmp_path)
