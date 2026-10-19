"""The four M1 quality measures (draft, confirm against the course rubric in T04).

1. Precision@K   : of the top-K ranked providers, the share labeled PotentialFraud = Yes
2. PR-AUC        : ranking quality over all providers (average precision)
3. Latency       : p50 / p95 time to score a batch of providers and return a ranked top-K
4. Robustness    : how much the top-K list changes when inputs get small random noise
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def rank_providers(provider_ids: pd.Series, scores: np.ndarray, limit: int = 20) -> pd.DataFrame:
    """Sort by score descending; ties broken by Provider ID so the order is deterministic."""
    ranked = (pd.DataFrame({"Provider": provider_ids.values, "score": scores})
              .sort_values(["score", "Provider"], ascending=[False, True])
              .head(limit).reset_index(drop=True))
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, provider_ids: pd.Series, k: int = 20) -> float:
    top = rank_providers(provider_ids, scores, limit=k)
    lookup = dict(zip(provider_ids.values, y_true))
    return float(np.mean([lookup[p] for p in top["Provider"]]))


def quality_metrics(y_true: np.ndarray, scores: np.ndarray, provider_ids: pd.Series) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision_at_20": precision_at_k(y_true, scores, provider_ids, 20),
        "precision_at_50": precision_at_k(y_true, scores, provider_ids, 50),
        "base_rate": float(np.mean(y_true)),
    }


def latency_ms(model, X: pd.DataFrame, provider_ids: pd.Series, repeats: int = 50, limit: int = 20) -> dict:
    """Time one full recommendation: score every provider, then sort and take the top K."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        rank_providers(provider_ids, model.predict_proba(X)[:, 1], limit)
        timings.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(timings, 50)), "p95_ms": float(np.percentile(timings, 95)),
            "providers_scored": int(len(X)), "note": "in-process scoring, not the network API"}


def robustness(model, X: pd.DataFrame, provider_ids: pd.Series, noise: float = 0.05,
               trials: int = 20, k: int = 20, seed: int = 0) -> dict:
    """Multiply every numeric input by (1 + N(0, noise)) and compare the top-K overlap."""
    rng = np.random.default_rng(seed)
    base = set(rank_providers(provider_ids, model.predict_proba(X)[:, 1], k)["Provider"])
    overlaps = []
    for _ in range(trials):
        noisy = X * (1 + rng.normal(0, noise, size=X.shape))
        top = set(rank_providers(provider_ids, model.predict_proba(noisy)[:, 1], k)["Provider"])
        overlaps.append(len(base & top) / k)
    return {"noise_level": noise, "trials": trials,
            "top20_overlap_mean": float(np.mean(overlaps)), "top20_overlap_min": float(np.min(overlaps))}
