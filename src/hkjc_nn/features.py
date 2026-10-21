"""Causal feature engineering.

Everything in this module obeys one rule: **a feature for race *r* may only use
information that was publicly available before race *r* started.**

Two classes of feature are produced.

*Same-race context* (draw, rating, declared weight, field strength) is known
from the racecard before the flag, so it is computed inside the race group and
is safe by construction.

*Historical* features (jockey win rate, trainer win rate, horse form,
distance/track suitability) are computed with a **strictly-prior-date**
expanding window.  Concretely, for a race on date *d*, only races with
``meeting_date < d`` contribute.  Races on the *same* day are excluded even
though the earlier ones would technically have finished already — being
conservative here costs a little signal and removes a whole class of subtle
leakage.

Rates with few observations are shrunk towards the global base rate
(empirical-Bayes style), because a jockey with three starts has a win rate
that is mostly noise.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import POST_RACE_ONLY_COLUMNS, TEXT_TIMESTAMP_COLUMNS

__all__ = [
    "FeatureConfig",
    "FeatureSet",
    "build_features",
    "prior_rate_table",
]


@dataclass(frozen=True)
class FeatureConfig:
    """Knobs for feature construction."""

    form_races: int = 5
    shrink_k: float = 10.0
    min_history: int = 1
    include_categorical: bool = True
    categorical_max_levels: int = 12
    text_columns: tuple[str, ...] = ("notes_text", "vet_report_text")

    #: Features that are deliberately left out.
    #:
    #: ``age`` is the horse's age **as of the collection date**, which is how
    #: HKJC publishes it.  For a 2018 race that value is the horse's age in
    #: 2026 — a future quantity, and one whose relationship to the outcome
    #: drifts with the calendar.  It is not a pre-race feature, and including
    #: it measurably damages the model: on the real data, dropping it takes
    #: the conditional logit's test log-loss from 2.50 back to 2.30.
    #:
    #: ``age_at_race`` is the correct, pre-race version and is kept.
    excluded_features: tuple[str, ...] = ("age",)


@dataclass
class FeatureSet:
    """A model-ready feature table."""

    frame: pd.DataFrame
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    medians_: dict[str, float] = field(default_factory=dict)

    @property
    def X(self) -> pd.DataFrame:
        cols = self.numeric_features + self.categorical_features
        return self.frame[cols]

    @property
    def y(self) -> np.ndarray:
        return self.frame["y"].to_numpy(dtype=float)

    @property
    def groups(self) -> np.ndarray:
        """Race identifiers, used to form listwise batches."""
        return self.frame["race_id"].to_numpy()

    def fill_missing(self, train_mask: np.ndarray | None = None) -> "FeatureSet":
        """Fill missing numeric values with medians taken from the training rows.

        This is deliberately *not* done inside :func:`build_features`.  Filling
        with a median computed over the whole dataset makes every feature
        depend on the whole dataset, which means appending new races changes
        the features of older ones — a form of transductive leakage, and one
        that a strict causality test will (rightly) fail.

        Call this with a mask over the **training** rows only.
        """
        mask = np.ones(len(self.frame), dtype=bool) if train_mask is None else np.asarray(train_mask)
        for col in self.numeric_features:
            series = pd.to_numeric(self.frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            reference = series[mask]
            value = float(reference.median()) if reference.notna().any() else 0.0
            self.medians_[col] = value
            self.frame[col] = series.fillna(value)
        return self

    def split_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(X, y, groups)`` with categoricals one-hot encoded.

        The dummy matrix is built from the categorical columns alone rather
        than by passing the whole frame to ``get_dummies``: depending on the
        pandas version, ``get_dummies`` may drop the original columns, which
        would take the numeric block with it.
        """
        X = self.X.copy()
        numeric = X[self.numeric_features].to_numpy(dtype=float)
        numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.categorical_features:
            return numeric, self.y, self.groups

        dummies = pd.get_dummies(
            X[self.categorical_features].astype(str), dummy_na=True, dtype=float
        )
        matrix = np.hstack([numeric, dummies.to_numpy(dtype=float)])
        return matrix, self.y, self.groups


# --------------------------------------------------------------------------------------
# Prior-history helpers
# --------------------------------------------------------------------------------------


def prior_rate_table(
    df: pd.DataFrame,
    keys: list[str],
    *,
    date_col: str = "meeting_date",
    value_col: str = "won",
    shrink_k: float = 10.0,
    base_rate: float | None = None,
) -> pd.DataFrame:
    """Cumulative win rate over strictly prior dates.

    Returns a table with the original keys plus ``*_prior_wins``,
    ``*_prior_starts`` and ``*_prior_rate`` (shrunk towards the global rate).

    ``base_rate`` is the shrinkage target.  It must be estimated from the
    **training** split only: if it is computed from the whole dataset, every
    historical feature silently depends on future races, and appending new
    data changes the features of older races.

    Example
    -------
    >>> rates = prior_rate_table(df, ["jockey_id"], base_rate=0.09)
    >>> rates["jockey_id_prior_rate"]  # win rate *before* each date
    """
    daily = (
        df.groupby(keys + [date_col], observed=True)[value_col]
        .agg(["sum", "size"])
        .reset_index()
        .sort_values(keys + [date_col])
    )
    grouper = daily.groupby(keys, observed=True, sort=False)
    daily["cum_wins"] = grouper["sum"].cumsum() - daily["sum"]
    daily["cum_starts"] = grouper["size"].cumsum() - daily["size"]

    if base_rate is None:
        base_rate = float(df[value_col].mean()) if len(df) else 0.0
    denom = daily["cum_starts"] + shrink_k
    rate = (daily["cum_wins"] + shrink_k * base_rate) / denom
    daily["rate"] = np.where(daily["cum_starts"] > 0, rate, base_rate)

    out = daily[keys + [date_col, "cum_wins", "cum_starts", "rate"]].copy()
    out = out.rename(
        columns={
            "cum_wins": f"{'_'.join(keys)}_prior_wins",
            "cum_starts": f"{'_'.join(keys)}_prior_starts",
            "rate": f"{'_'.join(keys)}_prior_rate",
        }
    )
    return out


def _attach_prior_rate(
    df: pd.DataFrame,
    keys: list[str],
    *,
    suffix: str,
    shrink_k: float,
    base_rate: float,
) -> pd.DataFrame:
    table = prior_rate_table(df, keys, shrink_k=shrink_k, base_rate=base_rate)
    out = df.merge(table, on=keys + ["meeting_date"], how="left")
    rate_col = f"{'_'.join(keys)}_prior_rate"
    if suffix:
        out = out.rename(columns={rate_col: f"{rate_col}_{suffix}"})
    return out


def _prior_form(df: pd.DataFrame, *, n: int = 5) -> pd.DataFrame:
    """Per-horse form and physical-condition features from prior races.

    Produces ``form_mean``, ``form_std``, ``form_win_rate``, ``days_since_last_run``,
    ``form_last_n`` (a compact string like ``"3-1-2-5-4"``), plus two
    condition features that the report asks for:

    ``declared_weight_change``
        The change in declared weight since the horse's previous run — the
        report's 磅重及變化.  A horse carrying noticeably less weight than
        last time is usually being fitted, and it is a genuine pre-race
        signal.

    ``sec_per_metre``
        The horse's average finishing speed over its last *n* runs, in seconds
        per metre.  This is the practical form of 分段時間(只用過往賽事):
        the raw sectional times are not published per horse historically, but
        the finishing time of the horse's own past runs is, and normalising it
        by distance makes runs comparable.
    """
    ordered = df.sort_values(["horse_id", "meeting_date", "race_id"])
    idx = ordered.index.to_numpy()

    # history[horse] holds (date, position, declared_weight, seconds, distance)
    history: dict[str, list[tuple]] = defaultdict(list)
    last_date: dict[str, pd.Timestamp] = {}

    means, stds, win_rates, days_since, strings = [], [], [], [], []
    wt_changes, speeds = [], []

    for row in ordered.itertuples(index=False):
        h = str(row.horse_id)
        d = pd.Timestamp(row.meeting_date)

        past = [rec for rec in history[h] if rec[0] < d]
        past_positions = [rec[1] for rec in past][-n:]

        if past_positions:
            arr = np.asarray(past_positions, dtype=float)
            means.append(float(arr.mean()))
            stds.append(float(arr.std(ddof=0)))
            win_rates.append(float((arr == 1).mean()))
        else:
            means.append(np.nan)
            stds.append(np.nan)
            win_rates.append(np.nan)

        prev = last_date.get(h)
        days_since.append(float((d - prev).days) if prev is not None else np.nan)
        last_date[h] = d

        strings.append("-".join(str(int(p)) for p in past_positions) if past_positions else "")

        # --- weight change vs the previous run ---------------------------------
        prev_weight = None
        for rec in reversed(past):
            if rec[2] is not None and not (rec[2] != rec[2]):  # not NaN
                prev_weight = rec[2]
                break
        current_weight = getattr(row, "declared_weight", None)
        if prev_weight is not None and current_weight is not None and current_weight == current_weight:
            wt_changes.append(float(current_weight) - float(prev_weight))
        else:
            wt_changes.append(np.nan)

        # --- average finishing speed over the last n runs -----------------------
        spd = [
            rec[3] / rec[4]
            for rec in past[-n:]
            if rec[3] is not None and rec[4] not in (None, 0) and rec[3] == rec[3]
        ]
        speeds.append(float(np.mean(spd)) if spd else np.nan)

        # Record the current result *after* the row has been scored, so it can
        # only ever be used by a strictly later race.  A runner that did not
        # finish has no position, and does not enter the form history.
        position = row.finish_position
        # The canonical column is ``win_time`` (seconds); some sources call it
        # ``finish_time``.
        raw_time = getattr(row, "win_time", None)
        if raw_time is None:
            raw_time = getattr(row, "finish_time", None)
        seconds = _parse_finish_seconds(raw_time)
        distance = getattr(row, "distance_m", None)
        if position == position and pd.notna(position):  # not NaN
            history[h].append(
                (d, int(position), current_weight, seconds, distance)
            )

    out = pd.DataFrame(
        {
            "form_mean": means,
            "form_std": stds,
            "form_win_rate": win_rates,
            "days_since_last_run": days_since,
            "form_last_n": strings,
            "declared_weight_change": wt_changes,
            "sec_per_metre": speeds,
        },
        index=idx,
    )
    out["form_last_n"] = out["form_last_n"].astype(str)
    return out


def _parse_finish_seconds(value) -> float | None:
    """Parse a finishing time like ``"1:24.27"`` into seconds."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return float(minutes) * 60.0 + float(seconds)
        return float(text)
    except ValueError:
        return None


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------


def build_features(
    races: pd.DataFrame,
    config: FeatureConfig | None = None,
    *,
    train_mask: np.ndarray | None = None,
) -> FeatureSet:
    """Build a causal feature table from a canonical race frame.

    Parameters
    ----------
    races
        Canonical frame (see :mod:`hkjc_nn.schema`).
    config
        Feature configuration.
    train_mask
        Boolean mask over the rows of ``races`` marking the **training**
        rows.  It is used to estimate the shrinkage base rate for the
        historical rates.  Leaving it unset makes every historical feature
        depend on the whole dataset, which breaks strict causality: append
        new races and the features of old races will change.
    """
    cfg = config or FeatureConfig()
    df = races.copy()

    missing = {"race_id", "meeting_date", "horse_id", "won", "finish_position"} - set(df.columns)
    if missing:
        raise ValueError(f"cannot build features, missing columns: {sorted(missing)}")

    df["meeting_date"] = pd.to_datetime(df["meeting_date"])
    # Keep a handle on the original row order so downstream code (the text
    # branch in particular) can join back to the raw frame.
    df["__row_id"] = np.arange(len(df))
    df = df.sort_values(["meeting_date", "race_id", "horse_id"]).reset_index(drop=True)

    # ---------------------------------------------------------------------------
    # 0. Drop anything that is only knowable after the race.
    #
    #    ``won`` and ``finish_position`` are *kept* for now: they are the
    #    outcomes of **past** races when the historical features are computed,
    #    which is exactly the information a pre-race model is allowed to see.
    #    ``win_time`` is kept for the same reason — the horse's own finishing
    #    times in *earlier* races are pre-race information.  None of them are
    #    ever features for the race they describe; the explicit safety check
    #    below enforces that.
    # ---------------------------------------------------------------------------
    outcome_cols = POST_RACE_ONLY_COLUMNS - {"won", "finish_position", "win_time"}
    dropped_post_race = [c for c in df.columns if c in outcome_cols]
    df = df.drop(columns=dropped_post_race, errors="ignore")

    # Text whose timestamp is after the race start is post-race commentary.
    dropped_late_text: list[str] = []
    if "race_start_time" in df.columns:
        race_start = pd.to_datetime(df["race_start_time"], errors="coerce")
        for text_col, ts_col in TEXT_TIMESTAMP_COLUMNS.items():
            if text_col not in df.columns:
                continue
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            late = ts.notna() & race_start.notna() & (ts > race_start)
            if late.any():
                df.loc[late, text_col] = np.nan
                dropped_late_text.append(f"{text_col}({int(late.sum())})")
    else:
        # Without a race start time we cannot prove the text was pre-race, so
        # we refuse to use it at all.
        for text_col in TEXT_TIMESTAMP_COLUMNS:
            if text_col in df.columns:
                df[text_col] = np.nan
                dropped_late_text.append(f"{text_col}(no race_start_time)")

    # ---------------------------------------------------------------------------
    # 1. Same-race context (all known from the racecard).
    # ---------------------------------------------------------------------------
    group = df.groupby("race_id", observed=True)

    # Rating-derived features are optional: a real extract may not carry a
    # rating at all, and the builder must not fall over when it is absent.
    if "rating" in df.columns:
        df["field_rating_mean"] = group["rating"].transform("mean")
        df["rating_vs_field_mean"] = df["rating"] - df["field_rating_mean"]
        df["rating_rank_in_field"] = group["rating"].rank(ascending=False, method="average")
    else:
        df["field_rating_mean"] = np.nan
        df["rating_vs_field_mean"] = np.nan
        df["rating_rank_in_field"] = np.nan
    df["rating_pct_in_field"] = df["rating_rank_in_field"] / group["race_id"].transform("size")

    if "declared_weight" in df:
        df["field_weight_mean"] = group["declared_weight"].transform("mean")
        df["weight_vs_field_mean"] = df["declared_weight"] - df["field_weight_mean"]
    else:
        df["weight_vs_field_mean"] = np.nan

    df["draw_pct"] = df["draw"] / df["num_runners"] if "draw" in df else np.nan
    df["is_favourite_by_rating"] = (df["rating_rank_in_field"] == 1).astype(int)

    # Distance bucket as an ordinal category.
    if "distance_m" in df:
        df["distance_bucket"] = pd.cut(
            df["distance_m"],
            bins=[0, 1200, 1600, 2000, 10_000],
            labels=["short", "sprinter", "middle", "staying"],
        ).astype(str)

    # ---------------------------------------------------------------------------
    # 2. Historical features (strictly prior dates).
    # ---------------------------------------------------------------------------
    # The shrinkage target for every rate is the overall win rate, estimated
    # from the training rows only.
    if train_mask is None:
        base_rate = float(df["won"].mean()) if len(df) else 0.0
    else:
        mask = np.asarray(train_mask, dtype=bool)
        if mask.shape[0] != len(df):
            raise ValueError("train_mask must align with the rows of `races`")
        base_rate = float(df.loc[mask, "won"].mean()) if mask.any() else float(df["won"].mean())

    df = _attach_prior_rate(df, ["jockey_id"], suffix="", shrink_k=cfg.shrink_k, base_rate=base_rate)
    df = _attach_prior_rate(df, ["trainer_id"], suffix="", shrink_k=cfg.shrink_k, base_rate=base_rate)
    df = _attach_prior_rate(
        df, ["jockey_id", "trainer_id"], suffix="", shrink_k=cfg.shrink_k, base_rate=base_rate
    )
    df = _attach_prior_rate(df, ["horse_id"], suffix="", shrink_k=cfg.shrink_k, base_rate=base_rate)

    if "distance_m" in df:
        df = _attach_prior_rate(
            df, ["horse_id", "distance_m"], suffix="", shrink_k=cfg.shrink_k, base_rate=base_rate
        )
        df = _attach_prior_rate(
            df, ["horse_id", "going_code"], suffix="", shrink_k=cfg.shrink_k, base_rate=base_rate
        )

    form = _prior_form(df, n=cfg.form_races)
    df = df.merge(form, left_index=True, right_index=True, how="left")

    # ---------------------------------------------------------------------------
    # 3. Assemble the feature list.
    # ---------------------------------------------------------------------------
    numeric_candidates = [
        "draw",
        "rating",
        "rating_change",
        "declared_weight",
        "weight_change",
        "handicap_weight",        "rating_vs_field_mean",
        "rating_rank_in_field",
        "rating_pct_in_field",
        "weight_vs_field_mean",
        "draw_pct",
        "is_favourite_by_rating",
        "field_rating_mean",
        "num_runners",
        "distance_m",
        "going_code",
        "prize_money",
        "form_mean",
        "form_std",
        "form_win_rate",
        "days_since_last_run",
        "declared_weight_change",
        "sec_per_metre",
        "age",
        "age_at_race",
        "vet_days_since",
        "vet_cleared",
        "jockey_id_prior_rate",
        "trainer_id_prior_rate",
        "jockey_id_trainer_id_prior_rate",
        "horse_id_prior_rate",
    ]
    if "horse_id_distance_m_prior_rate" in df:
        numeric_candidates += [
            "horse_id_distance_m_prior_rate",
            "horse_id_distance_m_prior_starts",
            "horse_id_going_code_prior_rate",
            "horse_id_going_code_prior_starts",
        ]
    numeric_candidates += [
        "jockey_id_prior_starts",
        "trainer_id_prior_starts",
        "jockey_id_trainer_id_prior_starts",
        "horse_id_prior_starts",
    ]

    categorical_candidates = [
        "race_class",
        "going",
        "going_code",
        "track",
        "distance_bucket",
        "sex",
        "country",
        "import_type",
        "venue",
    ]

    numeric = [c for c in numeric_candidates if c in df.columns]
    categorical = [c for c in categorical_candidates if c in df.columns] if cfg.include_categorical else []

    excluded = [c for c in cfg.excluded_features if c in numeric + categorical]
    if excluded:
        numeric = [c for c in numeric if c not in excluded]
        categorical = [c for c in categorical if c not in excluded]

    # Safety net: an outcome column must never become a feature, no matter
    # how the candidate lists above are edited.
    leaked = (set(numeric) | set(categorical)) & POST_RACE_ONLY_COLUMNS
    if leaked:
        raise AssertionError(f"post-race columns would leak into the features: {sorted(leaked)}")

    # Group rare categorical levels into "__other__" so one-off classes do not
    # explode the design matrix.
    for col in categorical:
        counts = df[col].astype(str).value_counts()
        keep = set(counts[counts >= max(2, len(df) // 500)].index)
        df[col] = df[col].astype(str).where(df[col].astype(str).isin(keep), "__other__")

    # Replace infinities; genuinely missing values stay missing and are
    # filled later from training statistics only (see FeatureSet.fill_missing).
    for col in numeric:
        series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df[col] = series

    out = pd.DataFrame(
        {
            "race_id": df["race_id"].to_numpy(),
            "meeting_date": df["meeting_date"].to_numpy(),
            "horse_id": df["horse_id"].to_numpy(),
            "row_id": df["__row_id"].to_numpy(),
            "y": df["won"].to_numpy(dtype=float),
            "finish_position": df["finish_position"].to_numpy(),
            "win_odds": df["win_odds"].to_numpy() if "win_odds" in df else np.nan,
            "pool_size": df["pool_size"].to_numpy() if "pool_size" in df else np.nan,
        }
    )
    # Keep the (leakage-scrubbed) text so the text branch can use it.
    for text_col in cfg.text_columns:
        if text_col in df.columns:
            out[text_col] = df[text_col].fillna("").astype(str).to_numpy()
    for col in numeric:
        out[col] = df[col].to_numpy()
    for col in categorical:
        out[col] = df[col].astype(str).to_numpy()

    fs = FeatureSet(frame=out, numeric_features=numeric, categorical_features=categorical)
    fs.leakage_audit = {  # type: ignore[attr-defined]
        "dropped_post_race_columns": dropped_post_race,
        "dropped_late_text": dropped_late_text,
    }
    return fs
