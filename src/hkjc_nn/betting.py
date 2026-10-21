"""Betting simulation: turn probabilities into money.

Design decisions that matter, and why
-------------------------------------

**Which odds.** The report insists on using *the odds available at the time
you bet*, not the final (SP) odds, because final odds embed information that
arrived after you committed.  The canonical frame's ``win_odds`` column is
therefore the **bet-time** odds; ``win_odds_open`` is the earlier quote.
Nothing here ever reaches for a post-race price.

**How the takeout is handled.** HKJC is pari-mutuel: the odds quoted for a
runner already embed the ~20% margin, because the pool is reduced by the
takeout before it is distributed.  So the naive calculation ``EV = c x o - 1``
is already correct, and the takeout does **not** need to be applied a second
time.  Applying ``(1 - takeout)`` on top of quoted HKJC odds would
double-count the margin.

Three payout models are provided:

``quoted``
    ``EV = c x o - 1``.  The correct HKJC convention.  Default.
``pari_mutuel``
    Models the effect of *our own stake* on the odds we receive.  When the
    pool size is known, the horse's share of the pool is backed out of the
    quoted odds, and our effective odds are computed exactly::

        P_i   = (1 - t) * P / o          # horse i's share of the pool
        ours  = (1 - t) * (P + s) / (P_i + s)

    As ``s -> 0`` this returns the quoted odds, so it degrades gracefully to
    ``quoted``.  This is the mechanism by which "betting too big lowers your
    own odds" is captured.
``effective``
    ``EV = c x o x (1 - t) - 1``.  For datasets whose quoted odds are *pre*
    takeout, and useful as a stress test.

**Staking.** Fractional Kelly (``f = f* x fraction``, default 1/4) with a
per-bet cap and a per-race exposure cap, because full Kelly on a noisy
probability estimate is ruinous.

**Bankroll.** Simulated chronologically so the bankroll that a bet is sized
against is the bankroll that actually existed at that point in time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass
class BetConfig:
    """Parameters of the staking simulation."""

    kelly_fraction: float = 0.25
    takeout: float = 0.225
    min_edge: float = 0.0
    max_odds: float | None = None
    payout_model: str = "quoted"
    max_bet_pct: float = 0.02
    max_race_pct: float = 0.05
    max_bets_per_race: int = 3
    start_bankroll: float = 10_000.0
    odds_column: str = "win_odds"
    odds_haircut: float = 0.0
    require_pool_size: bool = False
    label: str = "default"


@dataclass
class BettingResult:
    """Outcome of a betting simulation."""

    bets: pd.DataFrame
    summary: dict = field(default_factory=dict)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


def effective_odds(
    quoted_odds: float,
    stake: float,
    *,
    payout_model: str,
    takeout: float,
    pool_size: float | None = None,
) -> float:
    """Effective decimal odds received for a stake of ``stake``.

    See the module docstring for the three conventions.
    """
    if payout_model == "effective":
        return float(quoted_odds) * (1.0 - takeout)

    if payout_model == "pari_mutuel" and pool_size and pool_size > 0 and stake > 0:
        t = takeout
        P = float(pool_size)
        horse_pool = (1.0 - t) * P / float(quoted_odds)
        return float((1.0 - t) * (P + stake) / (horse_pool + stake))

    return float(quoted_odds)


def kelly_fraction(p: float, odds: float) -> float:
    """Full Kelly stake fraction for a binary bet at decimal odds ``odds``.

    ``f* = (p x odds - 1) / (odds - 1)``.  Zero when the bet has no edge.
    """
    if odds <= 1.0:
        return 0.0
    edge = p * odds - 1.0
    if edge <= 0:
        return 0.0
    return float(edge / (odds - 1.0))


def simulate_bets(
    predictions: pd.DataFrame,
    *,
    config: BetConfig | None = None,
    date_column: str = "meeting_date",
    race_column: str = "race_id",
    prob_column: str = "p_model",
) -> BettingResult:
    """Run a chronological betting simulation.

    Parameters
    ----------
    predictions
        Frame with one row per runner, containing at least
        ``race_id``, ``meeting_date``, the blended probability and the
        bet-time odds.
    config
        Staking configuration.
    prob_column
        Column holding the blended probability ``c_i`` used for the bet.
    """
    cfg = config or BetConfig()
    df = predictions.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    required = {race_column, date_column, prob_column, cfg.odds_column, "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"simulate_bets is missing columns: {sorted(missing)}")

    df = df.sort_values([date_column, race_column, prob_column], ascending=[True, True, False]).reset_index(
        drop=True
    )

    bankroll = float(cfg.start_bankroll)
    rows: list[dict] = []
    equity: list[dict] = []

    for race_id, race in df.groupby(race_column, sort=False):
        race_date = race[date_column].iloc[0]
        race_start_bankroll = bankroll
        race_staked = 0.0
        race_bets = 0

        pool = float(race["pool_size"].iloc[0]) if "pool_size" in race.columns and race["pool_size"].notna().any() else None

        for row in race.itertuples(index=False):
            if race_bets >= cfg.max_bets_per_race:
                break
            c = float(getattr(row, prob_column))
            odds = float(getattr(row, cfg.odds_column))
            if not np.isfinite(c) or not np.isfinite(odds) or odds <= 1.0:
                continue
            if c >= 1.0:
                continue
            # Probability estimates in the extreme tail are the least
            # reliable, so a cap on the price is a standard guard.
            if cfg.max_odds is not None and odds > cfg.max_odds:
                continue

            # The published HKJC price is the *starting price* — the pool's
            # final price, set after betting closed.  A bettor placing money
            # earlier faces a slightly worse price, so the haircut lets you
            # ask "does the edge survive a few percent of slippage?".
            if cfg.odds_haircut:
                odds = odds * (1.0 - cfg.odds_haircut)

            edge = c * odds - 1.0
            if edge <= cfg.min_edge:
                continue

            f_star = kelly_fraction(c, odds)
            if f_star <= 0:
                continue

            stake = bankroll * f_star * cfg.kelly_fraction
            stake = min(stake, bankroll * cfg.max_bet_pct)
            stake = min(stake, bankroll * cfg.max_race_pct - race_staked)
            if stake <= 0:
                continue

            eff_odds = effective_odds(
                odds,
                stake,
                payout_model=cfg.payout_model,
                takeout=cfg.takeout,
                pool_size=pool,
            )
            won = bool(getattr(row, "y") > 0.5)
            payout = stake * eff_odds if won else 0.0
            profit = payout - stake

            bankroll += profit
            race_staked += stake
            race_bets += 1

            rows.append(
                {
                    "race_id": race_id,
                    "meeting_date": race_date,
                    "horse_id": getattr(row, "horse_id", None),
                    "odds": odds,
                    "effective_odds": eff_odds,
                    "p_model": c,
                    "edge": edge,
                    "kelly_full": f_star,
                    "stake": stake,
                    "won": won,
                    "payout": payout,
                    "profit": profit,
                    "bankroll_after": bankroll,
                }
            )

        equity.append(
            {
                "race_id": race_id,
                "meeting_date": race_date,
                "bankroll": bankroll,
                "staked": race_staked,
                "return": bankroll - race_start_bankroll + race_staked,
                "n_bets": race_bets,
            }
        )

    bets = pd.DataFrame(rows)
    equity_df = pd.DataFrame(equity)
    summary = _summarise(bets, equity_df, cfg)
    return BettingResult(bets=bets, summary=summary, equity_curve=equity_df)


def _summarise(bets: pd.DataFrame, equity: pd.DataFrame, cfg: BetConfig) -> dict:
    if bets.empty:
        return {
            "n_bets": 0,
            "total_staked": 0.0,
            "total_payout": 0.0,
            "profit": 0.0,
            "roi": float("nan"),
            "hit_rate": float("nan"),
            "avg_odds": float("nan"),
            "final_bankroll": cfg.start_bankroll,
            "max_drawdown": 0.0,
            "profit_factor": float("nan"),
            "bets_per_race": 0.0,
        }

    staked = float(bets["stake"].sum())
    payout = float(bets["payout"].sum())
    profit = float(bets["profit"].sum())
    wins = int(bets["won"].sum())

    if len(equity):
        bankroll = equity["bankroll"].to_numpy()
        running_max = np.maximum.accumulate(bankroll)
        drawdown = float(np.max(running_max - bankroll) / max(running_max.max(), 1.0))
    else:
        drawdown = 0.0

    gross_win = float(bets.loc[bets["won"], "profit"].sum())
    gross_loss = float(-bets.loc[~bets["won"], "profit"].sum())

    # Predicted versus realised expected value.
    #
    # ``expected_profit`` is what the model's own probabilities promised:
    # sum over bets of (c x odds - 1) x stake.  Comparing it with the realised
    # profit is the cleanest way to tell genuine edge from over-confidence:
    # if realised profit is far *below* expected, the blended probability is
    # over-stated; if they agree, the edge is real.
    expected_profit = float(((bets["p_model"] * bets["effective_odds"] - 1.0) * bets["stake"]).sum())

    return {
        "n_bets": len(bets),
        "n_races_bet": int(bets["race_id"].nunique()),
        "total_staked": staked,
        "total_payout": payout,
        "profit": profit,
        "roi": profit / staked if staked > 0 else float("nan"),
        "roi_on_bankroll": profit / cfg.start_bankroll,
        "expected_profit": expected_profit,
        "ev_ratio": profit / expected_profit if abs(expected_profit) > 1e-9 else float("nan"),
        "hit_rate": wins / len(bets),
        "avg_odds": float(bets["effective_odds"].mean()),
        "avg_edge": float(bets["edge"].mean()),
        "final_bankroll": float(bets["bankroll_after"].iloc[-1]),
        "max_drawdown": drawdown,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "bets_per_race": len(bets) / max(bets["race_id"].nunique(), 1),
    }


def yearly_roi(bets: pd.DataFrame) -> pd.DataFrame:
    """ROI broken down by year — the report's "is it stable" check."""
    if bets.empty:
        return pd.DataFrame(columns=["year", "n_bets", "staked", "profit", "roi"])
    df = bets.copy()
    df["year"] = pd.to_datetime(df["meeting_date"]).dt.year
    grouped = df.groupby("year").agg(
        n_bets=("stake", "size"),
        staked=("stake", "sum"),
        profit=("profit", "sum"),
        hit_rate=("won", "mean"),
    )
    grouped["roi"] = grouped["profit"] / grouped["staked"].replace(0, np.nan)
    return grouped.reset_index()


def bet_calibration(bets: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Did the probabilities behind the bets hold up?

    For each price bucket this compares what the model *claimed* with what
    actually happened:

    * ``claimed_p`` vs ``actual_win_rate`` — is the blended probability honest?
    * ``claimed_ev`` vs ``realised_ev`` — does the expected value survive?

    This is the most important table in the whole project.  A strategy that
    shows a positive ROI while ``realised_ev`` is far below ``claimed_ev`` is
    not making money from information; it is getting lucky on a stream of
    bets whose expected value was never there.
    """
    if bets.empty:
        return pd.DataFrame(
            columns=[
                "bucket",
                "n_bets",
                "claimed_p",
                "actual_win_rate",
                "avg_odds",
                "claimed_ev",
                "realised_ev",
                "profit",
            ]
        )

    df = bets.copy()
    df["bucket"] = pd.qcut(df["p_model"].rank(method="first"), n_bins, labels=False, duplicates="drop")
    df["claimed_ev"] = df["p_model"] * df["effective_odds"] - 1.0
    df["realised_ev"] = df["won"] * df["effective_odds"] - 1.0

    grouped = df.groupby("bucket", observed=True).agg(
        n_bets=("stake", "size"),
        claimed_p=("p_model", "mean"),
        actual_win_rate=("won", "mean"),
        avg_odds=("effective_odds", "mean"),
        claimed_ev=("claimed_ev", "mean"),
        realised_ev=("realised_ev", "mean"),
        profit=("profit", "sum"),
    )
    grouped["bucket"] = grouped.index
    return grouped.reset_index(drop=True)
