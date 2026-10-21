"""Canonical schema for HKJC race data.

Every loader (synthetic, CSV, future API adapter) must produce the same wide
"one row per runner per race" frame.  Keeping a single canonical table means
the modelling code never needs to know where the data came from.

Leakage policy
--------------
Anything that was not publicly known *before the race started* must never be
used as a feature.  Two mechanisms enforce this:

1.  Columns whose name is listed in :data:`POST_RACE_ONLY_COLUMNS` are outcome
    columns.  They are used as labels/evaluation targets, never as features.
2.  Free-text columns carry their own ``*_timestamp`` companion column.  The
    feature builder drops any text whose timestamp is after the race start.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Column groups
# --------------------------------------------------------------------------------------

#: Identity of the race.
RACE_KEYS = ["race_id", "meeting_date"]

#: Static race context (known before the race).
RACE_CONTEXT = [
    "venue",
    "race_no",
    "race_class",
    "track",
    "going",
    "going_code",
    "distance_m",
    "num_runners",
    "prize_money",
    "race_start_time",
]

#: Static horse / runner context (known before the race).
RUNNER_CONTEXT = [
    "horse_id",
    "horse_name",
    "sex",
    "age",
    "country",
    "draw",
    "rating",
    "rating_change",
    "declared_weight",
    "weight_change",
    "jockey_id",
    "jockey_name",
    "trainer_id",
    "trainer_name",
    "saddlecloth",
]

#: Outcome columns.  NEVER features.
OUTCOME_COLUMNS = [
    "finish_position",
    "won",
    "beaten_margin",
    "win_time",
    "section_times",
]

#: Market data available before the race.
MARKET_COLUMNS = [
    "win_odds_open",
    "win_odds",
    "pool_size",
    "place_odds",
]

#: Pre-race free text with an accompanying timestamp column.
TEXT_COLUMNS = [
    "notes_text",
    "vet_report_text",
    "racecard_text",
]

#: Timestamp companion for each text column.
TEXT_TIMESTAMP_COLUMNS = {
    "notes_text": "notes_timestamp",
    "vet_report_text": "vet_timestamp",
    "racecard_text": "racecard_timestamp",
}

#: Post-race-only columns that a naive loader might carry in.
POST_RACE_ONLY_COLUMNS = set(OUTCOME_COLUMNS) | {"post_race_rating", "official_result_flag"}

#: Columns required in every canonical frame.
REQUIRED_COLUMNS = (
    RACE_KEYS
    + RACE_CONTEXT
    + RUNNER_CONTEXT
    + ["finish_position", "won", "win_odds"]
)

#: Columns that are allowed to be missing (filled with NaN / defaults).
OPTIONAL_COLUMNS = (
    MARKET_COLUMNS
    + TEXT_COLUMNS
    + list(TEXT_TIMESTAMP_COLUMNS.values())
    + ["days_since_last_run", "form_last_5", "rating_vs_field_mean"]
)

ALL_COLUMNS = list(dict.fromkeys(REQUIRED_COLUMNS + OPTIONAL_COLUMNS))


@dataclass(frozen=True)
class DataQualityReport:
    """Summary of what a loader produced."""

    n_races: int
    n_runners: int
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    missing_columns: tuple[str, ...] = ()
    duplicate_runners: int = 0
    overround_median: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        lines = [
            f"races={self.n_races}  runners={self.n_runners}",
        ]
        if self.date_min is not None:
            lines.append(f"dates={self.date_min.date()} .. {self.date_max.date()}")
        if self.missing_columns:
            lines.append(f"missing columns: {', '.join(self.missing_columns)}")
        if self.duplicate_runners:
            lines.append(f"duplicate (race_id, horse_id) rows: {self.duplicate_runners}")
        if self.overround_median is not None:
            lines.append(f"median overround: {self.overround_median:.3f}")
        lines.extend(f"WARN: {w}" for w in self.warnings)
        return "\n".join(lines)


def validate_frame(df: pd.DataFrame, *, strict: bool = False) -> DataQualityReport:
    """Check a frame against the canonical schema.

    Missing columns are reported but only raise in ``strict`` mode, so that a
    partially populated real-world extract can still be inspected before being
    rejected.
    """
    missing = tuple(c for c in REQUIRED_COLUMNS if c not in df.columns)
    if strict and missing:
        raise ValueError(f"canonical frame is missing required columns: {missing}")

    warnings: list[str] = []
    if missing:
        warnings.append(f"missing required columns: {', '.join(missing)}")

    n_races = 0
    n_runners = len(df)
    date_min = date_max = None
    dupes = 0
    overround_median = None

    if "race_id" in df.columns:
        n_races = df["race_id"].nunique()
    if {"race_id", "horse_id"}.issubset(df.columns):
        dupes = int(df.duplicated(subset=["race_id", "horse_id"]).sum())
        if dupes:
            warnings.append(f"{dupes} duplicated (race_id, horse_id) rows")
    if "meeting_date" in df.columns:
        dates = pd.to_datetime(df["meeting_date"], errors="coerce")
        if dates.notna().any():
            date_min, date_max = dates.min(), dates.max()
    if "win_odds" in df.columns:
        odds = pd.to_numeric(df["win_odds"], errors="coerce")
        ok = odds > 1.0
        if ok.any():
            per_race = (
                pd.DataFrame({"race_id": df.loc[ok, "race_id"], "r": 1.0 / odds[ok]})
                .groupby("race_id")["r"]
                .sum()
            )
            overround_median = float((per_race - 1.0).median())

    return DataQualityReport(
        n_races=n_races,
        n_runners=n_runners,
        date_min=date_min,
        date_max=date_max,
        missing_columns=missing,
        duplicate_runners=dupes,
        overround_median=overround_median,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------------------
# Market probability helpers
# --------------------------------------------------------------------------------------


def raw_implied(odds: pd.Series | np.ndarray) -> np.ndarray:
    """Raw (un-devigged) implied probabilities ``1/odds``."""
    arr = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(arr > 1.0, 1.0 / arr, np.nan)


def overround(odds: pd.Series | np.ndarray) -> float:
    """Sum of implied probabilities minus 1.  0.20 means a 20% takeout."""
    r = raw_implied(odds)
    r = r[~np.isnan(r)]
    return float(r.sum() - 1.0) if r.size else float("nan")


def devig_basic(odds: pd.Series | np.ndarray) -> np.ndarray:
    """Proportional ("basic") devig: ``pi_i = r_i / sum(r)``.

    Assumes the takeout is spread evenly over runners, which is a reasonable
    first-order approximation but does not correct for favourite-longshot bias.
    """
    r = raw_implied(odds)
    total = np.nansum(r)
    if not np.isfinite(total) or total <= 0:
        return np.full_like(r, np.nan)
    return r / total


def devig_shin(
    odds: pd.Series | np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> np.ndarray:
    """Shin (1993) devig, which corrects for favourite-longshot bias.

    Solves for the insider-trading parameter ``z`` such that the resulting
    probabilities sum to one, with longer-shot horses carrying proportionally
    less margin than favourites.

    Reference: Shin, H. S. (1993), *Measuring the incidence of insider trading in
    a market for state-contingent claims*, Economic Journal.
    """
    r = raw_implied(odds)
    valid = ~np.isnan(r)
    if not valid.any():
        return np.full_like(r, np.nan)

    rv = r[valid]
    total = rv.sum()
    if total <= 0:
        return np.full_like(r, np.nan)

    def probs(z: float) -> np.ndarray:
        z = min(max(z, 0.0), 1.0 - 1e-12)
        inner = z * z + 4.0 * (1.0 - z) * (rv * rv) / total
        return (np.sqrt(inner) - z) / (2.0 * (1.0 - z))

    lo, hi = 0.0, 1.0 - 1e-10
    f_lo = probs(lo).sum() - 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = probs(mid).sum() - 1.0
        if abs(f_mid) < tol:
            lo = hi = mid
            break
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    out = np.full_like(r, np.nan)
    out[valid] = probs(lo)
    return out


DEVG_METHODS = {"basic": devig_basic, "shin": devig_shin}


def devig(odds: pd.Series | np.ndarray, method: str = "shin") -> np.ndarray:
    """Remove the bookmaker margin from a set of decimal odds."""
    try:
        fn = DEVG_METHODS[method]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown devig method {method!r}; expected one of {sorted(DEVG_METHODS)}") from exc
    return fn(odds)
