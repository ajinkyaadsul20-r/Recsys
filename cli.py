"""CLI: recsys {train,recommend,similar,serve}."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence

from . import __version__
from .config import load_config
from .service import RecommenderService
from .train import train


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recsys", description=__doc__)
    p.add_argument("--version", action="version", version=f"recsys {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Generate data, tune, evaluate and save the best model")
    t.add_argument("--config", default=None)

    r = sub.add_parser("recommend", help="Top-N items for a user")
    r.add_argument("user_id", type=int)
    r.add_argument("-k", type=int, default=10)
    r.add_argument("--artifacts", default="artifacts")

    s = sub.add_parser("similar", help="Items similar to an item")
    s.add_argument("item_id", type=int)
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--artifacts", default="artifacts")

    sv = sub.add_parser("serve", help="Start the REST API (needs the [api] extra)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--artifacts", default="artifacts")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = _parser().parse_args(argv)
    if a.command == "train":
        meta = train(load_config(a.config))
        print(json.dumps(meta["leaderboard"], indent=2))
    elif a.command == "recommend":
        print(json.dumps(RecommenderService(a.artifacts).recommend(a.user_id, a.k), indent=2))
    elif a.command == "similar":
        print(json.dumps(RecommenderService(a.artifacts).similar(a.item_id, a.k), indent=2))
    elif a.command == "serve":
        import uvicorn

        os.environ["RECSYS_ARTIFACTS_DIR"] = a.artifacts
        uvicorn.run("recsys.api:app", host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
