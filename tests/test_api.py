import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from recsys import api  # noqa: E402


@pytest.fixture()
def client(trained_dir, monkeypatch):
    monkeypatch.setattr(api, "ARTIFACTS_DIR", str(trained_dir))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_recommend(client):
    r = client.get("/recommend/3?k=5")
    assert r.status_code == 200 and len(r.json()["items"]) == 5


def test_recommend_cold_start(client):
    assert client.get("/recommend/999999").json()["cold_start"] is True


def test_k_validation(client):
    assert client.get("/recommend/3?k=0").status_code == 422


def test_similar_and_404(client):
    assert client.get("/similar/2?k=3").status_code == 200
    assert client.get("/similar/999999").status_code == 404
