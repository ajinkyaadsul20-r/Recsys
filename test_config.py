from pathlib import Path

import pytest

from recsys.config import Config, load_config


def test_default_config_loads():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert set(cfg.models) == {"popularity", "item_knn", "als"}
    assert cfg.data.n_users > 0


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("nope.yaml")


def test_validation():
    with pytest.raises(ValueError):
        Config(k=0, models={"als": {}}).validate()
    with pytest.raises(ValueError):
        Config(models={"mystery": {}}).validate()
    with pytest.raises(ValueError):
        Config().validate()
