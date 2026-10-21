"""Serving logic shared by the CLI and API: recommendations with cold-start fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .train import BUNDLE_FILE, METADATA_FILE


class RecommenderService:
    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        art = Path(artifacts_dir)
        if not (art / BUNDLE_FILE).exists():
            raise FileNotFoundError(f"No trained model in '{art}'. Run `recsys train` first.")
        self.bundle = joblib.load(art / BUNDLE_FILE)
        self.metadata: dict[str, Any] = json.loads((art / METADATA_FILE).read_text())
        self.R = self.bundle.interactions
        self.items = self.bundle.items
        self.n_users, self.n_items = self.R.shape

    def _describe(self, idx: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "item_id": int(i),
                "category": str(self.items.at[int(i), "category"]),
                "score": round(float(s), 5),
            }
            for i, s in zip(idx, scores, strict=True)
        ]

    def recommend(self, user_id: int, k: int = 10) -> dict[str, Any]:
        if k < 1:
            raise ValueError("k must be >= 1")
        k = min(k, self.n_items)
        known = 0 <= user_id < self.n_users and self.R[user_id].nnz > 0
        if known:
            scores = self.bundle.model.score(np.array([user_id]))[0].astype(np.float64)
            scores[self.R[user_id].indices] = -np.inf  # never re-recommend seen items
        else:  # cold start: global popularity
            scores = self.bundle.popularity.pop.astype(np.float64).copy()
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return {
            "user_id": user_id,
            "cold_start": not known,
            "strategy": self.bundle.model_name if known else "popularity",
            "items": self._describe(top, scores[top]),
        }

    def similar(self, item_id: int, k: int = 10) -> dict[str, Any]:
        if not 0 <= item_id < self.n_items:
            raise KeyError(f"Unknown item_id {item_id}")
        try:
            idx, sims = self.bundle.model.similar_items(item_id, k)
        except NotImplementedError as exc:
            raise ValueError(str(exc)) from exc
        return {"item_id": item_id, "similar": self._describe(idx, sims)}
