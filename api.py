"""FastAPI app. Run: uvicorn recsys.api:app --host 0.0.0.0 --port 8000"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from . import __version__
from .service import RecommenderService

ARTIFACTS_DIR = os.getenv("RECSYS_ARTIFACTS_DIR", "artifacts")
_state: dict[str, RecommenderService] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["svc"] = RecommenderService(ARTIFACTS_DIR)
    yield
    _state.clear()


app = FastAPI(
    title="recsys API",
    version=__version__,
    description="Top-N recommendations with cold-start fallback and similar-item lookup.",
    lifespan=lifespan,
)


def _svc() -> RecommenderService:
    if "svc" not in _state:
        raise HTTPException(503, "Model not loaded")
    return _state["svc"]


@app.get("/health")
def health() -> dict[str, str]:
    _svc()
    return {"status": "ok", "version": __version__}


@app.get("/model")
def model_info() -> dict[str, Any]:
    return _svc().metadata


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
    return _svc().recommend(user_id, k)


@app.get("/similar/{item_id}")
def similar(item_id: int, k: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
    try:
        return _svc().similar(item_id, k)
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0])) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
