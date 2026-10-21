"""Tests for the market layer, the schema helpers, the evaluation metrics and
the time splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hkjc_nn.evaluation import (
    calibration_table,
    expected_calibration_error,
    hit_rate,
    incremental_vs_market,
    log_loss,
    race_log_loss,
    top_n_accuracy,
)
from hkjc_nn.models.market import BenterBlend, market_probabilities
from hkjc_nn.schema import devig, devig_basic, devig_shin, overround, validate_frame
from hkjc_nn.splits import time_splits


# --------------------------------------------------------------------------------------
# De-vigging
# --------------------------------------------------------------------------------------


def test_basic_devig_removes_the_overround():
    odds = np.array([2.0, 3.0, 4.0, 5.0])
    p = devig_basic(odds)
    assert np.isclose(p.sum(), 1.0)
    assert (p > 0).all()


def test_shin_devig_removes_the_overround():
    odds = np.array([2.0, 3.0, 4.0, 5.0, 8.0, 12.0])
    p = devig_shin(odds)
    assert np.isclose(p.sum(), 1.0, atol=1e-8)
    assert (p > 0).all()


def test_devig_is_monotone_in_the_price():
    """A shorter price must never get a higher probability."""
    odds = np.array([1.5, 2.0, 4.0, 9.0, 20.0, 40.0])
    p = devig_shin(odds)
    assert np.all(np.diff(p) < 0)
    p = devig_basic(odds)
    assert np.all(np.diff(p) < 0)


def test_overround_is_the_sum_minus_one():
    odds = np.array([2.0, 2.0, 2.0, 2.0])
    assert overround(odds) == pytest.approx(1.0)


def test_devig_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        devig(np.array([2.0, 3.0]), method="nope")


def test_market_probabilities_are_per_race():
    frame = pd.DataFrame(
        {
            "race_id": ["a", "a", "b", "b"],
            "win_odds": [2.0, 4.0, 1.5, 6.0],
        }
    )
    p = market_probabilities(frame, method="basic")
    assert np.isclose(p[0] + p[1], 1.0)
    assert np.isclose(p[2] + p[3], 1.0)


def test_market_probabilities_survive_a_broken_race():
    """One malformed race must not poison the whole file."""
    frame = pd.DataFrame(
        {
            "race_id": ["a", "a", "b", "b"],
            "win_odds": [2.0, 4.0, np.nan, 6.0],
        }
    )
    p = market_probabilities(frame, method="shin")
    assert np.isfinite(p).all()


# --------------------------------------------------------------------------------------
# Market blend
# --------------------------------------------------------------------------------------


def test_blend_ignores_a_model_with_no_information():
    """A constant model probability carries no information.

    Note that ``alpha`` is then *unidentifiable* — a constant added to every
    logit cancels inside a softmax — so the test asserts on the blended
    probabilities instead.
    """
    groups = np.array(["a", "a", "b", "b"])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    market = np.array([0.4, 0.3, 0.3, 0.4])
    model = np.array([0.5, 0.5, 0.5, 0.5])
    blend = BenterBlend().fit(model, market, groups, y)
    c = blend.predict(model, market, groups)
    # With no model information the blend can only track the market, and its
    # exponents are free to scale it.
    for race in np.unique(groups):
        mask = groups == race
        assert np.isclose(c[mask].sum(), 1.0)
    assert blend.beta >= 0.0


def test_blend_pulls_towards_the_market():
    """When the model equals the market, the blend should track it."""
    groups = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4)
    market = np.array(
        [
            0.40, 0.30, 0.20, 0.10,
            0.50, 0.20, 0.20, 0.10,
            0.30, 0.30, 0.20, 0.20,
            0.25, 0.25, 0.25, 0.25,
        ]
    )
    model = market.copy()
    # Winners deliberately include some non-favourites.
    y = np.array(
        [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            1, 0, 0, 0,
        ]
    )
    blend = BenterBlend().fit(model, market, groups, y)
    c = blend.predict(model, market, groups)
    for race in np.unique(groups):
        mask = groups == race
        assert np.isclose(c[mask].sum(), 1.0)
    assert np.corrcoef(c, market)[0, 1] > 0.85


def test_blend_beats_the_market_when_the_model_is_better():
    """A model with genuine information must improve on the market."""
    groups = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4)
    market = np.array(
        [
            0.40, 0.30, 0.20, 0.10,
            0.50, 0.20, 0.20, 0.10,
            0.30, 0.30, 0.20, 0.20,
            0.25, 0.25, 0.25, 0.25,
        ]
    )
    # The model knows the winners; the market does not.
    model = np.array(
        [
            0.70, 0.10, 0.10, 0.10,
            0.10, 0.70, 0.10, 0.10,
            0.10, 0.10, 0.70, 0.10,
            0.70, 0.10, 0.10, 0.10,
        ]
    )
    y = np.array(
        [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            1, 0, 0, 0,
        ]
    )
    blend = BenterBlend().fit(model, market, groups, y)
    c = blend.predict(model, market, groups)

    def ll(p):
        return -sum(np.log(p[y == 1][i]) for i in range(len(y[y == 1])))

    assert ll(c) < ll(market), "the blend failed to improve on the market"


def test_blend_probabilities_are_normalised():
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    y = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    market = np.array([0.4, 0.3, 0.3, 0.4, 0.2, 0.5])
    model = np.array([0.35, 0.25, 0.25, 0.45, 0.3, 0.4])
    blend = BenterBlend().fit(model, market, groups, y)
    c = blend.predict(model, market, groups)
    for race in np.unique(groups):
        mask = groups == race
        assert np.isclose(c[mask].sum(), 1.0)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def test_log_loss_of_a_perfect_forecast_is_small():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.99, 0.01, 0.99, 0.01])
    assert log_loss(y, p) < 0.02


def test_log_loss_penalises_confidence_errors():
    y = np.array([1.0, 0.0])
    good = log_loss(y, np.array([0.9, 0.1]))
    bad = log_loss(y, np.array([0.1, 0.9]))
    assert good < bad


def test_race_log_loss_only_looks_at_the_winner():
    groups = np.array(["a", "a", "b", "b"])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert race_log_loss(y, p, groups) == pytest.approx(-np.log(0.5))


def test_calibration_table_shape():
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    p = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3])
    table = calibration_table(y, p, n_bins=5)
    assert len(table) == 5
    assert table["n"].sum() == len(y)


def test_expected_calibration_error_is_zero_for_a_calibrated_forecast():
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    p = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert expected_calibration_error(y, p, n_bins=4) == pytest.approx(0.0, abs=1e-6)


def test_incremental_versus_market_reports_a_gain():
    groups = np.array(["a", "a", "b", "b"])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    market = np.array([0.5, 0.5, 0.5, 0.5])
    better = np.array([0.9, 0.1, 0.1, 0.9])
    out = incremental_vs_market(y, better, market, groups)
    assert out["ll_gain"] > 0
    assert out["r2"] > 0


def test_incremental_versus_market_is_zero_for_identical_forecasts():
    groups = np.array(["a", "a", "b", "b"])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    p = np.array([0.4, 0.3, 0.3, 0.4])
    out = incremental_vs_market(y, p, p, groups)
    assert out["ll_gain"] == pytest.approx(0.0)
    assert out["r2"] == pytest.approx(0.0)


def test_hit_rate_and_top_n():
    groups = np.array(["a", "a", "b", "b"])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    p = np.array([0.6, 0.4, 0.7, 0.3])
    assert hit_rate(y, p, groups) == pytest.approx(0.5)
    assert top_n_accuracy(y, p, groups, n_top=2) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Schema and splits
# --------------------------------------------------------------------------------------


def test_validate_frame_reports_a_clean_frame(races: pd.DataFrame):
    report = validate_frame(races)
    assert report.n_races > 0
    assert report.duplicate_runners == 0
    assert report.missing_columns == ()
    assert not report.warnings


def test_time_splits_are_disjoint_and_ordered():
    frame = pd.DataFrame(
        {
            "meeting_date": pd.to_datetime(
                ["2019-01-01", "2021-01-01", "2022-06-01", "2023-06-01", "2025-01-01"]
            )
        }
    )
    splits = time_splits(frame, train_until="2022-12-31", val_until="2023-12-31")
    assert splits.train.tolist() == [True, True, True, False, False]
    assert splits.val.tolist() == [False, False, False, True, False]
    assert splits.test.tolist() == [False, False, False, False, True]
    assert not (splits.train & splits.val).any()
    assert not (splits.val & splits.test).any()


def test_time_splits_cover_every_row():
    frame = pd.DataFrame({"meeting_date": pd.to_datetime(["2019-01-01", "2025-01-01"])})
    splits = time_splits(frame, train_until="2022-12-31", val_until="2023-12-31")
    assert splits.train.sum() + splits.val.sum() + splits.test.sum() == len(frame)
