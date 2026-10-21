"""Recommenders sharing one interface: fit(R) then score(users) -> dense score matrix."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp


class Recommender(ABC):
    name = "base"

    @abstractmethod
    def fit(self, R: sp.csr_matrix) -> Recommender: ...

    @abstractmethod
    def score(self, users: np.ndarray) -> np.ndarray:
        """Dense (len(users), n_items) relevance scores."""

    def similar_items(self, item: int, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(f"{self.name} does not support item similarity")


def _cosine_top(vectors: np.ndarray, item: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=1) + 1e-12
    sims = (vectors @ vectors[item]) / (norms * norms[item])
    sims[item] = -np.inf
    k = min(k, len(sims) - 1)
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return top, sims[top]


class Popularity(Recommender):
    """Most-interacted items. The baseline every model must beat."""

    name = "popularity"

    def fit(self, R: sp.csr_matrix) -> Popularity:
        self.pop = np.asarray(R.sum(axis=0)).ravel().astype(np.float64)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.pop, (len(users), 1))


class ItemKNN(Recommender):
    """Item-item cosine similarity, keeping only each item's top-N neighbours."""

    name = "item_knn"

    def __init__(self, neighbors: int = 50) -> None:
        self.neighbors = neighbors

    def fit(self, R: sp.csr_matrix) -> ItemKNN:
        self.R = R.tocsr().astype(np.float32)
        norms = np.sqrt(np.asarray(self.R.multiply(self.R).sum(axis=0)).ravel()) + 1e-12
        Rn = self.R @ sp.diags(1.0 / norms)
        S = (Rn.T @ Rn).toarray()
        np.fill_diagonal(S, 0.0)
        n = min(self.neighbors, S.shape[0] - 1)
        drop = np.argpartition(-S, n - 1, axis=1)[:, n:]
        np.put_along_axis(S, drop, 0.0, axis=1)
        self.S = S.astype(np.float32)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.asarray(self.R[users] @ self.S)

    def similar_items(self, item: int, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        sims = self.S[item].copy()
        k = min(k, len(sims) - 1)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return top, sims[top]


class ALS(Recommender):
    """Implicit-feedback matrix factorisation (Hu, Koren & Volinsky 2008) via
    alternating least squares. Observed entries get confidence 1 + alpha."""

    name = "als"

    def __init__(
        self,
        factors: int = 32,
        reg: float = 1.0,
        alpha: float = 20.0,
        iterations: int = 15,
        seed: int = 0,
    ) -> None:
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed

    def _solve(self, R: sp.csr_matrix, Y: np.ndarray) -> np.ndarray:
        f = self.factors
        base = Y.T @ Y + self.reg * np.eye(f)
        X = np.zeros((R.shape[0], f))
        for u in range(R.shape[0]):
            s, e = R.indptr[u], R.indptr[u + 1]
            if s == e:
                continue
            Yu = Y[R.indices[s:e]]
            conf = self.alpha * R.data[s:e]
            A = base + (Yu.T * conf) @ Yu
            b = Yu.T @ (1.0 + conf)
            X[u] = np.linalg.solve(A, b)
        return X

    def fit(self, R: sp.csr_matrix) -> ALS:
        R = R.tocsr().astype(np.float64)
        Rt = R.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.V = rng.normal(0, 0.1, (R.shape[1], self.factors))
        self.U = np.zeros((R.shape[0], self.factors))
        for _ in range(self.iterations):
            self.U = self._solve(R, self.V)
            self.V = self._solve(Rt, self.U)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.U[users] @ self.V.T

    def similar_items(self, item: int, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        return _cosine_top(self.V, item, k)


def build_model(name: str, seed: int = 0, **params) -> Recommender:
    if name == "popularity":
        return Popularity()
    if name == "item_knn":
        return ItemKNN(**params)
    if name == "als":
        return ALS(seed=seed, **params)
    raise ValueError(f"Unknown model '{name}'")
