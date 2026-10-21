"""Evaluation metrics.

The report is emphatic about the ordering of importance, and this module
follows it:

1.  **Log-loss** is the primary metric.  It is the proper scoring rule for
    probabilistic forecasts and the only metric that rewards both sharpness
    and calibration.
2.  **Calibration** — "does a horse the model gives 20% actually win 20% of
    the time?" — is the second metric, and the one that separates a
    well-behaved model from a merely lucky one.
3.  **Incremental value versus the market** is the third, and it is the one
    that actually answers the project's question.  A model can have a better
    log-loss than the market and still be unprofitable, if its edge lives on
    favourites whose prices already reflect it.
4.  Hit rate and top-N accuracy are reported as **auxiliary** metrics only.
    A model can pick the winner 60% of the time and lose money, because
    favourites pay little.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "log_loss",
    "race_log_loss",
    "calibration_table",
    "expected_calibration_error",
    "brier_score",
    "hit_rate",
    "top_n_accuracy",
    "incremental_vs_market",
    "evaluate",
]

EPS = 1e-15


def _prepare(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    return y, p


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean per-runner negative log-likelihood (the headline metric)."""
    y, p = _prepare(y, p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def race_log_loss(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> float:
    """Mean per-race negative log-likelihood of the winner's probability.

    This is the log-likelihood of the *event* "the model correctly picked the
    winner of this race", which is what a bettor actually cares about, and it
    is the quantity Benter's ``R^2`` measure is built on.
    """
    y, p = _prepare(y, p)
    groups = np.asarray(groups)
    total, n = 0.0, 0
    for race in np.unique(groups):
        mask = groups == race
        yy, pp = y[mask], p[mask]
        if yy.sum() < 1:
            continue
        winner_idx = int(np.argmax(yy))
        total -= float(np.log(pp[winner_idx]))
        n += 1
    return total / max(n, 1)


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted probability vs observed win frequency."""
    y, p = _prepare(y, p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append(
            {
                "bin": b,
                "lo": edges[b],
                "hi": edges[b + 1],
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()) if mask.any() else np.nan,
                "observed_frequency": float(y[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean absolute gap between predicted and observed frequency."""
    table = calibration_table(y, p, n_bins)
    total = table["n"].sum()
    if total == 0:
        return float("nan")
    gaps = (table["mean_predicted"] - table["observed_frequency"]).abs().fillna(0.0)
    return float((gaps * table["n"]).sum() / total)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the win indicator."""
    y, p = _prepare(y, p)
    return float(np.mean((p - y) ** 2))


def hit_rate(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of races where the model's highest-probability runner won.

    Auxiliary metric only — see the module docstring.
    """
    y, p = _prepare(y, p)
    groups = np.asarray(groups)
    hits, n = 0, 0
    for race in np.unique(groups):
        mask = groups == race
        yy = y[mask]
        if yy.sum() < 1:
            continue
        hits += int(yy[int(np.argmax(p[mask]))] > 0.5)
        n += 1
    return hits / max(n, 1)


def top_n_accuracy(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_top: int = 2) -> float:
    """Fraction of races where the winner was in the model's top ``n_top``."""
    y, p = _prepare(y, p)
    groups = np.asarray(groups)
    hits, n = 0, 0
    for race in np.unique(groups):
        mask = groups == race
        yy = y[mask]
        if yy.sum() < 1:
            continue
        order = np.argsort(-p[mask])[:n_top]
        hits += int(yy[order].sum() > 0.5)
        n += 1
    return hits / max(n, 1)


def incremental_vs_market(
    y: np.ndarray,
    p_model: np.ndarray,
    p_market: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Value of the model relative to the market.

    ``r2`` follows Benter's convention: the proportional reduction in
    log-likelihood versus using the market probability directly.  A positive
    value means the model extracts information the market does not have.
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)

    ll_model = race_log_loss(y, p_model, groups)
    ll_market = race_log_loss(y, p_market, groups)

    total_model = 0.0
    total_market = 0.0
    n = 0
    for race in np.unique(groups):
        mask = groups == race
        yy = y[mask]
        if yy.sum() < 1:
            continue
        w = int(np.argmax(yy))
        total_model -= float(np.log(np.clip(p_model[mask][w], EPS, 1.0)))
        total_market -= float(np.log(np.clip(p_market[mask][w], EPS, 1.0)))
        n += 1

    return {
        "ll_model": ll_model,
        "ll_market": ll_market,
        "ll_gain": ll_market - ll_model,
        "r2": 1.0 - (total_model / total_market) if total_market > 0 else float("nan"),
        "n_races": n,
    }


def evaluate(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    *,
    market_p: np.ndarray | None = None,
    n_bins: int = 10,
) -> dict:
    """Full metric bundle for one model."""
    metrics = {
        "log_loss": log_loss(y, p),
        "race_log_loss": race_log_loss(y, p, groups),
        "brier": brier_score(y, p),
        "ece": expected_calibration_error(y, p, n_bins),
        "hit_rate": hit_rate(y, p, groups),
        "top2_accuracy": top_n_accuracy(y, p, groups, 2),
    }
    if market_p is not None:
        metrics.update(
            {f"vs_market_{k}": v for k, v in incremental_vs_market(y, p, market_p, groups).items()}
        )
    return metrics
