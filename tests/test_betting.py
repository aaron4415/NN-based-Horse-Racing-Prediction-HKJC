"""Tests for the betting simulation.

The Kelly formula and the payout models are where a subtle mistake turns a
losing strategy into an apparently winning one, so they are pinned down
against closed-form values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hkjc_nn.betting import BetConfig, bet_calibration, effective_odds, kelly_fraction, simulate_bets


def test_kelly_fraction_matches_the_closed_form():
    # p = 0.5, odds = 2.0  ->  f* = (0.5*2 - 1) / (2 - 1) = 0
    assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0)
    # p = 0.6, odds = 2.0  ->  f* = (1.2 - 1) / 1 = 0.2
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    # No edge -> no bet.
    assert kelly_fraction(0.4, 2.0) == 0.0
    assert kelly_fraction(0.9, 1.05) == 0.0


def test_kelly_is_zero_below_even_money():
    assert kelly_fraction(0.49, 2.0) == 0.0


def test_effective_odds_quoted_mode_is_the_quote():
    assert effective_odds(5.0, 100.0, payout_model="quoted", takeout=0.225) == 5.0


def test_effective_odds_effective_mode_applies_the_takeout():
    out = effective_odds(5.0, 100.0, payout_model="effective", takeout=0.20)
    assert out == pytest.approx(4.0)


def test_pari_mutuel_impact_reduces_odds_as_the_stake_grows():
    """Betting bigger into a pool must lower the odds you receive."""
    small = effective_odds(10.0, 100.0, payout_model="pari_mutuel", takeout=0.225, pool_size=1_000_000)
    large = effective_odds(10.0, 100_000.0, payout_model="pari_mutuel", takeout=0.225, pool_size=1_000_000)
    assert large < small
    assert small <= 10.0


def test_pari_mutuel_falls_back_to_quoted_without_a_pool():
    out = effective_odds(10.0, 100.0, payout_model="pari_mutuel", takeout=0.225, pool_size=None)
    assert out == 10.0


def _frame(rows):
    return pd.DataFrame(rows)


def test_only_positive_ev_bets_are_placed():
    frame = _frame(
        [
            # c * odds = 0.4 * 2.0 = 0.8  -> negative EV, no bet
            {"race_id": "r1", "meeting_date": "2024-01-01", "horse_id": "H1", "y": 1,
             "win_odds": 2.0, "pool_size": 1e6, "p_model": 0.4},
            # c * odds = 0.6 * 2.0 = 1.2  -> positive EV, bet
            {"race_id": "r2", "meeting_date": "2024-01-02", "horse_id": "H2", "y": 0,
             "win_odds": 2.0, "pool_size": 1e6, "p_model": 0.6},
        ]
    )
    result = simulate_bets(frame, config=BetConfig())
    assert len(result.bets) == 1
    assert result.bets.iloc[0]["race_id"] == "r2"


def test_winning_bet_pays_the_odds():
    frame = _frame(
        [{"race_id": "r1", "meeting_date": "2024-01-01", "horse_id": "H1", "y": 1,
          "win_odds": 4.0, "pool_size": 1e6, "p_model": 0.5}]
    )
    result = simulate_bets(frame, config=BetConfig(kelly_fraction=1.0))
    bet = result.bets.iloc[0]
    assert bet["stake"] > 0
    assert bet["payout"] == pytest.approx(bet["stake"] * 4.0)
    assert bet["profit"] == pytest.approx(bet["stake"] * 3.0)


def test_losing_bet_costs_the_stake():
    frame = _frame(
        [{"race_id": "r1", "meeting_date": "2024-01-01", "horse_id": "H1", "y": 0,
          "win_odds": 4.0, "pool_size": 1e6, "p_model": 0.5}]
    )
    result = simulate_bets(frame, config=BetConfig())
    bet = result.bets.iloc[0]
    assert bet["payout"] == 0.0
    assert bet["profit"] == pytest.approx(-bet["stake"])


def test_bankroll_changes_by_the_profit():
    frame = _frame(
        [
            {"race_id": "r1", "meeting_date": "2024-01-01", "horse_id": "H1", "y": 1,
             "win_odds": 4.0, "pool_size": 1e6, "p_model": 0.5},
            {"race_id": "r2", "meeting_date": "2024-01-02", "horse_id": "H2", "y": 0,
             "win_odds": 4.0, "pool_size": 1e6, "p_model": 0.5},
        ]
    )
    result = simulate_bets(frame, config=BetConfig(start_bankroll=1000.0))
    total_profit = result.bets["profit"].sum()
    assert result.summary["final_bankroll"] == pytest.approx(1000.0 + total_profit)


def test_per_bet_cap_is_respected():
    frame = _frame(
        [{"race_id": "r1", "meeting_date": "2024-01-01", "horse_id": "H1", "y": 0,
          "win_odds": 2.0, "pool_size": 1e6, "p_model": 0.9}]
    )
    cfg = BetConfig(kelly_fraction=1.0, max_bet_pct=0.01, start_bankroll=10_000.0)
    result = simulate_bets(frame, config=cfg)
    assert result.bets.iloc[0]["stake"] <= 100.0 + 1e-9


def test_max_odds_cap_filters_longshots():
    frame = _frame(
        [
            {"race_id": "r1", "meeting_date": "2024-01-01", "horse_id": "H1", "y": 0,
             "win_odds": 50.0, "pool_size": 1e6, "p_model": 0.5},
            {"race_id": "r2", "meeting_date": "2024-01-02", "horse_id": "H2", "y": 0,
             "win_odds": 3.0, "pool_size": 1e6, "p_model": 0.5},
        ]
    )
    result = simulate_bets(frame, config=BetConfig(max_odds=10.0))
    assert set(result.bets["race_id"]) == {"r2"}


def test_bets_are_ordered_by_date():
    frame = _frame(
        [
            {"race_id": "r2", "meeting_date": "2024-03-01", "horse_id": "H2", "y": 1,
             "win_odds": 2.0, "pool_size": 1e6, "p_model": 0.6},
            {"race_id": "r1", "meeting_date": "2024-01-01", "horse_id": "H1", "y": 0,
             "win_odds": 2.0, "pool_size": 1e6, "p_model": 0.6},
        ]
    )
    result = simulate_bets(frame, config=BetConfig())
    dates = result.bets["meeting_date"].tolist()
    assert dates == sorted(dates)


def test_empty_result_is_handled():
    frame = _frame(
        [{"race_id": "r1", "meeting_date": "2024-01-01", "horse_id": "H1", "y": 1,
          "win_odds": 2.0, "pool_size": 1e6, "p_model": 0.1}]
    )
    result = simulate_bets(frame, config=BetConfig())
    assert result.bets.empty
    assert result.summary["n_bets"] == 0
    assert np.isnan(result.summary["roi"])


def test_bet_calibration_reports_claimed_versus_realised():
    frame = _frame(
        [
            {"race_id": "r1", "meeting_date": "2024-01-01", "horse_id": "H1", "y": 1,
             "win_odds": 4.0, "pool_size": 1e6, "p_model": 0.5},
            {"race_id": "r2", "meeting_date": "2024-01-02", "horse_id": "H2", "y": 0,
             "win_odds": 4.0, "pool_size": 1e6, "p_model": 0.5},
        ]
    )
    result = simulate_bets(frame, config=BetConfig())
    table = bet_calibration(result.bets)
    assert not table.empty
    assert {"claimed_p", "actual_win_rate", "claimed_ev", "realised_ev"} <= set(table.columns)
