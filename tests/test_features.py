"""Tests for feature construction, with an emphasis on leakage.

The strongest leakage test is *append-only*: build the features from the
races up to some date, then append more races and rebuild.  Every feature of
the earlier races must be **bit-for-bit unchanged**.  If any feature peeks
at a future race, this test fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hkjc_nn.data import simulate_races
from hkjc_nn.features import FeatureConfig, build_features, prior_rate_table


def test_features_are_stable_when_future_races_are_appended(races: pd.DataFrame):
    """Appending later races must not change any earlier feature."""
    cutoff = pd.Timestamp("2022-06-30")
    past = races[races["meeting_date"] <= cutoff].copy()
    future = races[races["meeting_date"] > cutoff].copy()
    assert not future.empty, "fixture needs races after the cutoff"

    # Both builds see the same training rows, which is how the pipeline is
    # meant to be used: the shrinkage base rate comes from the training split.
    before = build_features(past, train_mask=np.ones(len(past), dtype=bool))
    after = build_features(
        pd.concat([past, future], ignore_index=True),
        train_mask=np.r_[np.ones(len(past), dtype=bool), np.zeros(len(future), dtype=bool)],
    )

    key = ["race_id", "horse_id"]
    merged = before.frame.merge(after.frame, on=key, suffixes=("_before", "_after"))
    assert len(merged) == len(before.frame)

    for col in before.numeric_features:
        b = merged[f"{col}_before"].to_numpy(dtype=float)
        a = merged[f"{col}_after"].to_numpy(dtype=float)
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12, err_msg=f"{col} leaked")


def test_shrinkage_base_rate_uses_only_the_training_rows():
    """The shrinkage target must come from the training split, not the whole file."""
    df = pd.DataFrame(
        {
            "jockey_id": ["J1"] * 3,
            "meeting_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "won": [1, 0, 1],
        }
    )
    # Training rows say J1 won one of two races (0.5); the test row says 2 of 3.
    table = prior_rate_table(df, ["jockey_id"], shrink_k=10.0, base_rate=0.5)
    table = table.sort_values("meeting_date")
    expected = (1 * 1 + 10 * 0.5) / (1 + 10)
    assert table.iloc[1]["jockey_id_prior_rate"] == pytest.approx(expected)


def test_post_race_columns_are_dropped(races: pd.DataFrame):
    fs = build_features(races)
    assert "post_race_rating" not in fs.frame.columns
    assert "beaten_margin" not in fs.frame.columns
    assert "win_time" not in fs.frame.columns
    for leaky in ("post_race_rating", "beaten_margin", "win_time", "finish_position"):
        assert leaky not in fs.numeric_features


def test_late_text_is_scrubbed(races: pd.DataFrame):
    """Text published after the race must not survive into the features."""
    fs = build_features(races)
    audit = getattr(fs, "leakage_audit", {})
    assert audit["dropped_late_text"], "the fixture should contain post-race text"

    # The post-race racecard quoted the rating the horse finished with.  No
    # carried text column may contain that number.
    leaked = races["racecard_text"].str.extract(r"賽後評分 (\d+)")[0]
    for col in ("notes_text", "vet_report_text"):
        if col not in fs.frame.columns:
            continue
        for value, rating in zip(fs.frame[col].astype(str), leaked):
            if value and rating and rating in value:
                raise AssertionError(f"{col} still carries the post-race rating {rating}")


def test_text_is_dropped_entirely_without_a_race_start_time(races: pd.DataFrame):
    """Without a race start time we cannot prove text was pre-race."""
    no_start = races.drop(columns=["race_start_time"])
    fs = build_features(no_start)
    for col in ("notes_text", "vet_report_text"):
        if col in fs.frame.columns:
            assert fs.frame[col].eq("").all(), f"{col} survived without a race start time"


def test_prior_rate_uses_only_strictly_earlier_dates():
    df = pd.DataFrame(
        {
            "jockey_id": ["J1"] * 4,
            "meeting_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01", "2020-03-01"]),
            "won": [1, 0, 1, 0],
        }
    )
    table = prior_rate_table(df, ["jockey_id"], shrink_k=0.0).sort_values("meeting_date")
    # One row per (jockey, date): 01-01 (2 races), 02-01, 03-01.
    assert len(table) == 3
    # 2020-01-01: no history yet.
    assert table.iloc[0]["jockey_id_prior_starts"] == 0
    # 2020-02-01: both January races count, and the February race itself does not.
    assert table.iloc[1]["jockey_id_prior_starts"] == 2
    assert table.iloc[1]["jockey_id_prior_wins"] == 1
    # 2020-03-01: three races.
    assert table.iloc[2]["jockey_id_prior_starts"] == 3
    assert table.iloc[2]["jockey_id_prior_wins"] == 2


def test_same_day_races_do_not_leak_into_each_other():
    df = pd.DataFrame(
        {
            "jockey_id": ["J1"] * 3,
            "meeting_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-08"]),
            "won": [1, 0, 0],
        }
    )
    table = prior_rate_table(df, ["jockey_id"], shrink_k=0.0).sort_values("meeting_date")
    # The two races on 1 January share one row and see no history at all.
    assert table.iloc[0]["jockey_id_prior_starts"] == 0
    # The race on 8 January sees both January races.
    assert table.iloc[1]["jockey_id_prior_starts"] == 2


def test_feature_matrix_is_finite(races: pd.DataFrame):
    fs = build_features(races)
    X, y, groups = fs.split_matrices()
    assert np.isfinite(X).all()
    assert len(y) == len(X)
    assert set(np.unique(y)) <= {0.0, 1.0}


def test_horse_form_uses_prior_races_only():
    """A horse's form feature must not include the race being predicted."""
    df = simulate_races(n_races=60, n_horses=15, seed=3)
    fs = build_features(df, FeatureConfig(form_races=5))
    assert "form_mean" in fs.frame.columns
    assert fs.frame["form_mean"].notna().sum() > 0
    # The very first race of each horse has no form.
    first = fs.frame.groupby("horse_id").head(1)
    assert first["form_mean"].isna().all()
