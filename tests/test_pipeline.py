"""End-to-end tests: the CSV adapter, the data loader and the full runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hkjc_nn.data import load_races, simulate_races
from hkjc_nn.data.csv_adapter import describe_csv, load_csv_races, write_canonical_csv
from hkjc_nn.experiments import ExperimentConfig, run_experiments
from hkjc_nn.schema import validate_frame


# --------------------------------------------------------------------------------------
# CSV adapter
# --------------------------------------------------------------------------------------


def test_csv_adapter_round_trip(tmp_path, races: pd.DataFrame):
    path = tmp_path / "canonical.csv"
    write_canonical_csv(races, path)
    loaded = load_csv_races(path)
    assert len(loaded) == len(races)
    assert loaded["race_id"].nunique() == races["race_id"].nunique()
    assert np.allclose(
        loaded.sort_values(["race_id", "horse_id"])["win_odds"].to_numpy(),
        races.sort_values(["race_id", "horse_id"])["win_odds"].to_numpy(),
    )


def test_csv_adapter_accepts_vendor_column_names(tmp_path):
    """A file with Chinese headers and unit-suffixed values still loads."""
    raw = pd.DataFrame(
        {
            "賽事日期": ["01/01/2023", "01/01/2023", "08/01/2023", "08/01/2023"],
            "場次": [1, 1, 1, 1],
            "馬名": ["A", "B", "C", "D"],
            "檔位": [1, 2, 1, 2],
            "途程": ["1200M", "1200M", "1400M", "1400M"],
            "磅重": ["540", "542", "545", "547"],
            "名次": [1, 2, 1, 3],
            "賠率": [3.5, 8.0, 2.1, 15.0],
            "騎師": ["J1", "J2", "J1", "J3"],
            "練馬師": ["T1", "T2", "T1", "T3"],
        }
    )
    path = tmp_path / "vendor.csv"
    raw.to_csv(path, index=False, encoding="utf-8-sig")

    frame = load_csv_races(path, date_format="%d/%m/%Y")
    assert len(frame) == 4
    assert frame["race_id"].nunique() == 2
    assert frame["distance_m"].tolist() == [1200, 1200, 1400, 1400]
    assert frame["declared_weight"].tolist() == [540.0, 542.0, 545.0, 547.0]
    assert frame["won"].tolist() == [1, 0, 1, 0]
    assert frame["num_runners"].tolist() == [2, 2, 2, 2]


def test_describe_csv_reports_what_is_missing(tmp_path, races: pd.DataFrame):
    path = tmp_path / "canonical.csv"
    write_canonical_csv(races, path)
    table = describe_csv(path)
    assert table.loc[table["canonical"] == "race_id", "present"].all()
    assert len(table) > 20


def test_load_races_dispatches_on_the_source(tmp_path, races: pd.DataFrame):
    path = tmp_path / "canonical.csv"
    write_canonical_csv(races, path)
    assert len(load_races(path)) == len(races)
    assert len(load_races("synthetic", n_races=10, n_horses=10)) > 0


def test_load_races_rejects_a_missing_file():
    with pytest.raises(FileNotFoundError):
        load_races("does-not-exist.csv")


# --------------------------------------------------------------------------------------
# Synthetic generator
# --------------------------------------------------------------------------------------


def test_synthetic_frame_is_valid():
    df = simulate_races(n_races=120, n_horses=30, seed=5)
    report = validate_frame(df)
    assert report.n_races == 120
    assert report.duplicate_runners == 0
    assert report.missing_columns == ()


def test_synthetic_race_ids_are_unique():
    df = simulate_races(n_races=200, n_horses=40, seed=5)
    # One race id per race, one row per runner.
    assert not df.duplicated(subset=["race_id", "horse_id"]).any()
    assert df.groupby("race_id").size().min() >= 2


def test_exactly_one_winner_per_race():
    df = simulate_races(n_races=120, n_horses=30, seed=5)
    winners = df.groupby("race_id")["won"].sum()
    assert (winners == 1).all()


def test_synthetic_market_has_an_overround():
    df = simulate_races(n_races=120, n_horses=30, seed=5)
    for _race, group in list(df.groupby("race_id"))[:20]:
        implied = (1.0 / group["win_odds"]).sum()
        assert implied > 1.0, "the synthetic market must carry a margin"


def test_synthetic_generator_is_deterministic():
    a = simulate_races(n_races=60, n_horses=20, seed=99)
    b = simulate_races(n_races=60, n_horses=20, seed=99)
    pd.testing.assert_frame_equal(a, b)


def test_post_race_traps_are_present_by_default():
    df = simulate_races(n_races=40, n_horses=15, seed=5)
    assert "post_race_rating" in df.columns
    assert "racecard_text" in df.columns
    assert "racecard_timestamp" in df.columns


# --------------------------------------------------------------------------------------
# Full runner
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def experiment_results():
    cfg = ExperimentConfig(
        synthetic_kwargs={
            "n_races": 350,
            "n_horses": 50,
            "seed": 11,
            "start_date": "2019-01-01",
            "end_date": "2023-12-31",
        },
        train_until="2021-12-31",
        val_until="2022-12-31",
        nn_epochs=3,
        models=("A", "B", "F"),
    )
    return run_experiments(cfg)


def test_runner_produces_metrics_for_every_model(experiment_results):
    metrics = experiment_results.metrics
    assert not metrics.empty
    assert set(experiment_results.metrics["model"]) == {"A", "B", "F"}
    # The runner reports out-of-sample splits only; training metrics would be
    # in-sample and meaningless.
    assert set(metrics["split"]) == {"val", "test"}
    assert metrics["log_loss"].notna().all()


def test_runner_separates_the_splits(experiment_results):
    metrics = experiment_results.metrics
    for split in ("val", "test"):
        rows = metrics[metrics["split"] == split]
        assert rows["n_rows"].nunique() == 1


def test_runner_produces_a_betting_summary(experiment_results):
    summary = experiment_results.betting_summary
    assert summary
    assert "roi" in summary
    assert "ev_ratio" in summary
    assert "max_drawdown" in summary


def test_runner_predictions_cover_the_test_split(experiment_results):
    preds = experiment_results.predictions
    assert "p_market" in preds.columns
    assert preds["p_blend"].notna().any()


def test_runner_is_deterministic():
    cfg = ExperimentConfig(
        synthetic_kwargs={"n_races": 200, "n_horses": 30, "seed": 3},
        train_until="2021-12-31",
        val_until="2022-12-31",
        nn_epochs=2,
        models=("A", "B"),
    )
    first = run_experiments(cfg)
    second = run_experiments(cfg)
    pd.testing.assert_frame_equal(first.metrics, second.metrics)


def test_null_market_produces_no_profit():
    """With an optimal market, nothing should show a profit.

    This is the harness's own self-check: if the betting simulation can
    manufacture money when there is nothing to find, none of its other
    numbers mean anything.
    """
    cfg = ExperimentConfig(
        synthetic_kwargs={
            "n_races": 400,
            "n_horses": 60,
            "seed": 21,
            "market_skill": 1.0,
            "market_specialist": 1.0,
            "market_condition": 1.0,
            "market_noise": 0.0,
        },
        train_until="2022-12-31",
        val_until="2023-12-31",
        nn_epochs=3,
        models=("A", "B", "F"),
    )
    results = run_experiments(cfg)
    test = results.metrics[results.metrics["split"] == "test"]
    assert (test["vs_market_r2"] < 0.05).all(), "a model beat an optimal market"
    assert results.betting_summary["roi"] < 0.10, "the simulation manufactured a profit"
