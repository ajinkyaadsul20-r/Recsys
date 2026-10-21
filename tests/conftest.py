"""Shared fixtures: a small, fast end-to-end training run."""

from __future__ import annotations

import pytest

from recsys.config import Config, DataConfig
from recsys.train import train


@pytest.fixture(scope="session")
def small_cfg_factory():
    def make(out):
        return Config(
            seed=1,
            k=10,
            artifacts_dir=out,
            data=DataConfig(n_users=300, n_items=120, n_categories=4, mean_interactions=25),
            models={
                "popularity": {},
                "item_knn": {"neighbors": [10, 20]},
                "als": {"factors": [4, 8], "reg": [10.0], "alpha": [5.0], "iterations": 8},
            },
        )

    return make


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory, small_cfg_factory):
    out = tmp_path_factory.mktemp("artifacts")
    train(small_cfg_factory(out))
    return out
