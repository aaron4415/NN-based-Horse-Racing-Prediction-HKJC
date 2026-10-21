"""Experiment runner for the A-F matrix.

The matrix from the report:

===== =============================== ==================================
Code  Model                           Purpose
===== =============================== ==================================
A     Conditional logit (tabular)    Minimum bar; outputs calibrated probs
B     LightGBM (tabular)             Strong tree baseline
C     NN (tabular)                   Does deep learning help at all?
D     NN (tabular + text)            Is text worth anything?
E     LightGBM (tabular + text)      Fair control: trees + text
F     Best of A-E + market odds       Can anything beat the market?
===== =============================== ==================================

D versus E is the decisive comparison in the NN story: if D does not clearly
beat E, the extra architectural complexity is not justified.  F is the money
question, and it is answered with the betting simulation rather than with
hit rate.

Two rules keep the experiment honest:

* Model selection for F uses the **validation** split only.  The test split is
  touched once, at the end.
* Every model takes an explicit seed, so the whole run is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .betting import BetConfig, bet_calibration, simulate_bets, yearly_roi
from .data import load_races
from .evaluation import calibration_table, evaluate
from .features import FeatureConfig, FeatureSet, build_features
from .models.conditional_logit import ConditionalLogit
from .models.lightgbm_model import LightGBMModel
from .models.market import BenterBlend, market_probabilities
from .schema import validate_frame
from .splits import TimeSplits, describe_splits, time_splits

__all__ = ["ExperimentConfig", "ExperimentResults", "run_experiments"]


@dataclass
class ExperimentConfig:
    """Everything the runner needs."""

    # --- data -----------------------------------------------------------------
    data_source: str = "synthetic"
    synthetic_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "n_races": 2500,
            "n_horses": 220,
            "seed": 42,
            "market_skill": 1.0,
            "market_specialist": 0.80,
            "market_condition": 0.0,
            "market_noise": 0.55,
            "signal_noise": 0.35,
        }
    )
    csv_kwargs: dict[str, Any] = field(default_factory=dict)

    # --- split ----------------------------------------------------------------
    train_until: str = "2022-12-31"
    val_until: str = "2023-12-31"

    # --- features --------------------------------------------------------------
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    text_dim: int = 12
    use_text: bool = True

    # --- models ----------------------------------------------------------------
    models: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
    nn_architecture: str = "field_transformer"  # or "listwise_mlp"
    nn_epochs: int = 40
    nn_hidden: tuple[int, ...] = (128, 64)
    seed: int = 42

    # --- betting ----------------------------------------------------------------
    betting: BetConfig = field(default_factory=BetConfig)
    devig_method: str = "basic"


@dataclass
class ExperimentResults:
    """Everything the runner produced."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    betting_summary: dict
    yearly: pd.DataFrame
    data_summary: str
    split_summary: str
    blend_summary: dict
    calibration: dict[str, pd.DataFrame] = field(default_factory=dict)
    bet_calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_names: list[str] = field(default_factory=list)
    coverage_shift: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        lines = ["=" * 78, "DATA", "-" * 78, self.data_summary, "", "SPLIT", "-" * 78, self.split_summary]
        lines += ["", "MODEL METRICS", "-" * 78]
        if not self.metrics.empty:
            lines.append(self.metrics.to_string(index=False))
        if self.blend_summary:
            lines += ["", "MARKET BLEND", "-" * 78]
            lines.append(
                f"base model = {self.blend_summary.get('base_model')}   "
                f"alpha = {self.blend_summary.get('alpha'):.3f}   "
                f"beta = {self.blend_summary.get('beta'):.3f}   "
                f"devig = {self.blend_summary.get('devig_method')}"
            )
        lines += ["", "BETTING SIMULATION (test split)", "-" * 78]
        if self.betting_summary:
            for key, value in self.betting_summary.items():
                lines.append(f"  {key:20s} {value}")
        else:
            lines.append("  (not run)")
        if not self.bet_calibration.empty:
            lines += ["", "BET CALIBRATION (claimed vs realised)", "-" * 78]
            lines.append(self.bet_calibration.to_string(index=False))
        if not self.yearly.empty:
            lines += ["", "YEARLY ROI", "-" * 78]
            lines.append(self.yearly.to_string(index=False))
        lines.append("=" * 78)
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _encode(fs: FeatureSet, splits: TimeSplits) -> tuple[np.ndarray, list[str]]:
    """One-hot encode categoricals using **training levels only**."""
    from sklearn.preprocessing import OneHotEncoder

    cat_cols = fs.categorical_features
    num_cols = fs.numeric_features
    frame = fs.frame

    if not cat_cols:
        return frame[num_cols].to_numpy(dtype=float), list(num_cols)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(frame.loc[splits.train, cat_cols].astype(str))
    dense = encoder.transform(frame[cat_cols].astype(str))
    names = [
        f"{col}__{value}"
        for col, values in zip(cat_cols, encoder.categories_)
        for value in list(values)
    ]

    matrix = np.hstack([frame[num_cols].to_numpy(dtype=float), dense])
    return matrix, list(num_cols) + names


def _build_model(code: str, cfg: ExperimentConfig):
    """Instantiate the model behind an experiment code."""
    if code == "A":
        return ConditionalLogit(l2=1.0)
    if code in {"B", "E"}:
        return LightGBMModel(random_state=cfg.seed)
    if code in {"C", "D"}:
        from .models.neural import FieldTransformer, ListwiseMLP

        if cfg.nn_architecture == "listwise_mlp":
            return ListwiseMLP(hidden_sizes=cfg.nn_hidden, epochs=cfg.nn_epochs, seed=cfg.seed)
        return FieldTransformer(
            hidden_sizes=cfg.nn_hidden,
            epochs=cfg.nn_epochs,
            seed=cfg.seed,
            d_model=64,
        )
    raise ValueError(f"unknown experiment code {code!r}")


def _fit_model(model, X_train, y_train, g_train, X_val, y_val, g_val):
    """Fit a model, passing validation data only to models that accept it."""
    if isinstance(model, ConditionalLogit):
        model.fit(X_train, y_train, g_train)
    else:
        model.fit(X_train, y_train, g_train, X_val=X_val, y_val=y_val, groups_val=g_val)


def _safe_market_probabilities(frame, groups, method: str) -> np.ndarray:
    """Devigged market probabilities, with graceful fallbacks."""
    p = market_probabilities(frame, odds_column="win_odds", method=method, groups=groups)
    if np.isnan(p).any():
        from .schema import devig_basic

        fallback = market_probabilities(frame, odds_column="win_odds", method="basic", groups=groups)
        p = np.where(np.isnan(p), fallback, p)
    # Any remaining gaps become uniform within the race.
    if np.isnan(p).any():
        p = np.where(np.isnan(p), 0.0, p)
        p = p / np.maximum(_group_sums(p, groups), 1e-12)
    return p


def _group_sums(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    for g in np.unique(groups):
        mask = groups == g
        out[mask] = values[mask].sum()
    return out


# --------------------------------------------------------------------------------------
# main runner
# --------------------------------------------------------------------------------------


def _coverage_shift(fs, splits) -> pd.DataFrame:
    """Compare each feature's availability between the training and test rows.

    A feature whose availability changes a lot over time is a **train/serve
    shift**: the model learns its coefficient from one population and applies
    it to another.  That is much more dangerous than plain missing data,
    because the model cannot tell that anything is wrong.

    ``age_at_race`` is the worked example — HKJC only publishes the age for
    horses imported from 2019, so it is present for ~10% of the training rows
    and ~36% of the test rows, and including it makes the model *worse*.
    """
    rows = []
    for col in fs.numeric_features:
        train_rate = float(fs.frame.loc[splits.train, col].notna().mean()) if splits.train.any() else 0.0
        test_rate = float(fs.frame.loc[splits.test, col].notna().mean()) if splits.test.any() else 0.0
        rows.append(
            {
                "feature": col,
                "train_coverage": train_rate,
                "test_coverage": test_rate,
                "shift": test_rate - train_rate,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.reindex(table["shift"].abs().sort_values(ascending=False).index)
    return table


def run_experiments(cfg: ExperimentConfig | None = None) -> ExperimentResults:
    """Run the full experiment matrix and return the results."""
    cfg = cfg or ExperimentConfig()

    # --- 1. data ------------------------------------------------------------------
    if cfg.data_source.lower() == "synthetic":
        races = load_races("synthetic", **cfg.synthetic_kwargs)
    else:
        races = load_races(cfg.data_source, **cfg.csv_kwargs)
    data_report = validate_frame(races)
    data_summary = data_report.summary()

    # --- 2. features ---------------------------------------------------------------
    # The split is decided on the raw frame (it only needs the meeting date),
    # so that the training mask can be handed to build_features.  The mask
    # keeps the shrinkage base rate — and therefore every historical feature
    # — strictly causal.
    raw_dates = pd.to_datetime(races["meeting_date"])
    train_mask = (raw_dates <= pd.Timestamp(cfg.train_until)).to_numpy()

    fs = build_features(races, cfg.feature_config, train_mask=train_mask)
    splits = time_splits(
        fs.frame, train_until=cfg.train_until, val_until=cfg.val_until, date_column="meeting_date"
    )
    split_summary = describe_splits(fs.frame, splits)

    # Missing values are filled with medians taken from the training rows only.
    # Filling from the whole dataset would make every feature depend on the
    # test split.
    fs.fill_missing(splits.train)

    X_all, feature_names = _encode(fs, splits)

    # Report any feature whose availability shifts between train and test.
    shift = _coverage_shift(fs, splits)
    shifted = shift[shift["shift"].abs() > 0.15] if not shift.empty else shift
    if not shifted.empty:
        print("\nWARNING: feature availability shifts between train and test:", flush=True)
        for _, r in shifted.iterrows():
            print(
                f"   {r['feature']:24s} train={r['train_coverage']:.0%}  "
                f"test={r['test_coverage']:.0%}  shift={r['shift']:+.0%}",
                flush=True,
            )
    y = fs.y
    groups = fs.groups

    # --- 3. text branch -------------------------------------------------------------
    text_matrix = np.zeros((len(fs.frame), 0))
    text_names: list[str] = []
    if cfg.use_text:
        from .text import build_text_matrix

        text_matrix, text_names = build_text_matrix(
            fs.frame,
            text_columns=tuple(cfg.feature_config.text_columns),
            fit_frame=fs.frame.loc[splits.train],
            n_components=cfg.text_dim,
        )

    # --- 4. market probabilities ----------------------------------------------------
    market_p = _safe_market_probabilities(fs.frame, groups, cfg.devig_method)

    # --- 5. run the models -----------------------------------------------------------
    metric_rows: list[dict] = []
    predictions = fs.frame[
        ["race_id", "meeting_date", "horse_id", "y", "win_odds", "pool_size"]
    ].copy()
    predictions["p_market"] = market_p

    model_probs: dict[str, dict[str, np.ndarray]] = {}
    calibration_tables: dict[str, pd.DataFrame] = {
        "market": calibration_table(y[splits.test], market_p[splits.test], n_bins=10)
    }

    for code in cfg.models:
        if code == "F":
            continue

        use_text = code in {"D", "E"} and text_matrix.shape[1] > 0

        X_train = X_all[splits.train]
        y_train = y[splits.train]
        g_train = groups[splits.train]
        X_val = X_all[splits.val]
        X_test = X_all[splits.test]

        if use_text:
            X_train = np.hstack([X_train, text_matrix[splits.train]])
            X_val = np.hstack([X_val, text_matrix[splits.val]])
            X_test = np.hstack([X_test, text_matrix[splits.test]])

        model = _build_model(code, cfg)
        _fit_model(model, X_train, y_train, g_train, X_val, y[splits.val], groups[splits.val])

        preds = {
            "val": model.predict_proba(X_val, groups[splits.val]),
            "test": model.predict_proba(X_test, groups[splits.test]),
        }
        model_probs[code] = preds

        calibration_tables[code] = calibration_table(
            y[splits.test], preds["test"], n_bins=10
        )

        for split_name, mask in (("val", splits.val), ("test", splits.test)):
            m = evaluate(
                y[mask], preds[split_name], groups[mask], market_p=market_p[mask]
            )
            m.update({"model": code, "split": split_name, "n_rows": int(mask.sum())})
            metric_rows.append(m)

    # --- 6. F: best model + market ---------------------------------------------------
    blend_summary: dict = {}
    betting_summary: dict = {}
    yearly = pd.DataFrame()
    bet_cal = pd.DataFrame()
    equity = pd.DataFrame()

    if "F" in cfg.models and model_probs:
        # Pick the model with the best *validation* log-loss, so the test split
        # is never used for model selection.
        val_scores = {
            code: next(
                r["log_loss"] for r in metric_rows if r["model"] == code and r["split"] == "val"
            )
            for code in model_probs
            if any(r["model"] == code and r["split"] == "val" for r in metric_rows)
        }
        best_code = min(val_scores, key=val_scores.get) if val_scores else None

        if best_code is not None:
            blend = BenterBlend(devig_method=cfg.devig_method)
            blend.fit(
                model_probs[best_code]["val"],
                market_p[splits.val],
                groups[splits.val],
                y[splits.val],
            )
            blend_summary = blend.summary()
            blend_summary["base_model"] = best_code

            c_test = blend.predict(
                model_probs[best_code]["test"], market_p[splits.test], groups[splits.test]
            )
            predictions.loc[splits.test, "p_model"] = model_probs[best_code]["test"]
            predictions.loc[splits.test, "p_blend"] = c_test

            m = evaluate(y[splits.test], c_test, groups[splits.test], market_p=market_p[splits.test])
            m.update({"model": "F", "split": "test", "n_rows": int(splits.test.sum())})
            metric_rows.append(m)

            bet_frame = predictions.loc[splits.test].copy()
            bet_frame["p_model"] = c_test
            result = simulate_bets(bet_frame, config=cfg.betting, prob_column="p_model")
            betting_summary = dict(result.summary)
            betting_summary["base_model"] = best_code
            betting_summary["label"] = cfg.betting.label
            yearly = yearly_roi(result.bets)
            bet_cal = bet_calibration(result.bets)
            equity = result.equity_curve
            calibration_tables["F (blend)"] = calibration_table(
                y[splits.test], c_test, n_bins=10
            )

    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        preferred = [
            "model",
            "split",
            "n_rows",
            "log_loss",
            "race_log_loss",
            "ece",
            "hit_rate",
            "top2_accuracy",
        ]
        preferred += [c for c in metrics.columns if c not in preferred]
        metrics = metrics[preferred].sort_values(["split", "model"]).reset_index(drop=True)

    return ExperimentResults(
        metrics=metrics,
        predictions=predictions,
        betting_summary=betting_summary,
        yearly=yearly,
        data_summary=data_summary,
        split_summary=split_summary,
        blend_summary=blend_summary,
        calibration=calibration_tables,
        bet_calibration=bet_cal,
        equity_curve=equity,
        feature_names=list(feature_names),
        coverage_shift=shift,
    )
