"""Training: tune each model on the validation item, report on the test item, then refit the
winner on ALL interactions for serving."""

from __future__ import annotations

import itertools
import json
import logging
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp

from . import __version__
from .config import Config
from .data import add_targets, generate_data, leave_last_out, to_matrix
from .evaluate import evaluate, hit_rate_curve
from .models import Popularity, build_model
from .plots import save_plots

log = logging.getLogger(__name__)
BUNDLE_FILE = "bundle.joblib"
METADATA_FILE = "metadata.json"


@dataclass
class Bundle:
    """Everything needed to serve recommendations."""

    model: Any
    popularity: Popularity
    interactions: sp.csr_matrix
    items: pd.DataFrame
    model_name: str


def _grid(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand {a: [1,2], b: 3} into [{a:1,b:3},{a:2,b:3}]."""
    if not spec:
        return [{}]
    keys = list(spec)
    values = [v if isinstance(v, list) else [v] for v in spec.values()]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def train(cfg: Config) -> dict[str, Any]:
    started = time.perf_counter()
    d = cfg.data
    interactions, items = generate_data(d, cfg.seed)
    split = leave_last_out(interactions, d.n_users, d.n_items)
    train_val_seen = split.train
    test_seen = add_targets(split.train, split.val_target)
    metric = f"ndcg@{cfg.k}"
    log.info("users=%d items=%d interactions=%d", d.n_users, d.n_items, len(interactions))

    results: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for name, spec in cfg.models.items():
        best: dict[str, Any] | None = None
        for params in _grid(spec):
            model = build_model(name, seed=cfg.seed, **params).fit(split.train)
            val, _ = evaluate(model, split.train, train_val_seen, split.val_target, cfg.k)
            if best is None or val[metric] > best["val"][metric]:
                best = {"params": params, "val": val}
        assert best is not None
        # Final eval: refit on train + validation items, test on the last item.
        full_train = add_targets(split.train, split.val_target)
        model = build_model(name, seed=cfg.seed, **best["params"]).fit(full_train)
        test, ranks = evaluate(model, full_train, test_seen, split.test_target, cfg.k)
        results.append(
            {
                "model": name,
                "params": best["params"],
                "val": best["val"],
                "test": test,
                "hit_curve": hit_rate_curve(ranks),
            }
        )
        fitted[name] = best["params"]
        log.info("%-11s val %s=%.4f | test %s=%.4f", name, metric, best["val"][metric],
                 metric, test[metric])

    results.sort(key=lambda r: r["test"][metric], reverse=True)
    winner = max(results, key=lambda r: r["val"][metric])  # choose on validation, not test
    log.info("Selected (by validation): %s", winner["model"])

    # Production refit on every interaction.
    all_m = to_matrix(interactions, d.n_users, d.n_items)
    final = build_model(winner["model"], seed=cfg.seed, **winner["params"]).fit(all_m)
    bundle = Bundle(final, Popularity().fit(all_m), all_m, items, winner["model"])

    out = cfg.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)
    save_plots(results, items, final, cfg.k, out / "plots")
    joblib.dump(bundle, out / BUNDLE_FILE)

    meta = {
        "package_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_seconds": round(time.perf_counter() - started, 1),
        "seed": cfg.seed,
        "k": cfg.k,
        "selected_model": winner["model"],
        "selection": f"validation {metric}",
        "n_users": d.n_users,
        "n_items": d.n_items,
        "n_interactions": int(len(interactions)),
        "leaderboard": [
            {"model": r["model"], "params": r["params"], "test": r["test"], "val": r["val"]}
            for r in results
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    (out / METADATA_FILE).write_text(json.dumps(meta, indent=2))
    return meta
