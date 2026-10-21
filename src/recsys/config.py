"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    n_users: int = 2000
    n_items: int = 800
    n_categories: int = 8
    mean_interactions: int = 30


@dataclass(frozen=True)
class Config:
    seed: int = 42
    k: int = 10
    artifacts_dir: Path = Path("artifacts")
    data: DataConfig = field(default_factory=DataConfig)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.data.n_users < 10 or self.data.n_items < 10:
            raise ValueError("need at least 10 users and 10 items")
        if self.data.n_categories < 1 or self.data.mean_interactions < 3:
            raise ValueError("invalid data settings")
        unknown = set(self.models) - {"popularity", "item_knn", "als"}
        if unknown:
            raise ValueError(f"unknown models: {sorted(unknown)}")
        if not self.models:
            raise ValueError("config must define at least one model")


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or "configs/default.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    raw["data"] = DataConfig(**raw.get("data", {}))
    if "artifacts_dir" in raw:
        raw["artifacts_dir"] = Path(raw["artifacts_dir"])
    raw["models"] = {k: (v or {}) for k, v in raw.get("models", {}).items()}
    cfg = Config(**raw)
    cfg.validate()
    return cfg
