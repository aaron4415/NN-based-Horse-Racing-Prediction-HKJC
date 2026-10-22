"""Feature coverage audit against the project's feature specification.

The report lists the features a racing model is supposed to use.  This module
checks, row by row, which of them are actually implemented, which are
available in the collected data, and which cannot be obtained from HKJC's
public pages at all.

Run it with::

    hjkc-nn coverage --data data/hkjc_races.csv
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["FEATURE_SPEC", "audit_features", "coverage_table"]


@dataclass(frozen=True)
class FeatureRequirement:
    """One line of the feature specification."""

    category: str
    name: str
    columns: tuple[str, ...]
    status: str  # "core" | "extra" | "unavailable"
    note: str = ""


#: The specification, in the order the report lists it.
FEATURE_SPEC: tuple[FeatureRequirement, ...] = (
    # ---- 賽事 -----------------------------------------------------------------
    FeatureRequirement("賽事", "場地 / venue", ("venue",), "core"),
    FeatureRequirement("賽事", "途程 / distance", ("distance_m",), "core"),
    FeatureRequirement("賽事", "班次 / race class", ("race_class",), "core"),
    FeatureRequirement("賽事", "場地狀況 / going", ("going", "going_code"), "core"),
    FeatureRequirement("賽事", "馬匹數目 / field size", ("num_runners",), "core"),
    FeatureRequirement("賽事", "跑道狀況 / track condition", ("track", "track_code"), "core"),
    FeatureRequirement("賽事", "獎金 / prize money", ("prize_money",), "extra"),
    FeatureRequirement("賽事", "賽事名 / race name", ("race_name",), "extra"),
    FeatureRequirement("賽事", "班次評分範圍 / class rating band", ("rating_band_low", "rating_band_high"), "extra"),
    # ---- 馬匹 -----------------------------------------------------------------
    FeatureRequirement("馬匹", "檔位 / draw", ("draw",), "core"),
    FeatureRequirement("馬匹", "評分 / rating", ("rating",), "core"),
    FeatureRequirement("馬匹", "評分變化 / rating change", ("rating_change",), "core"),
    FeatureRequirement("馬匹", "體重 / declared weight", ("declared_weight",), "core"),
    FeatureRequirement("馬匹", "體重變化 / weight change", ("declared_weight_change",), "core"),
    FeatureRequirement("馬匹", "年齡 / age", ("age_at_race",), "core"),
    FeatureRequirement("馬匹", "距上次出賽日數 / days since last run", ("days_since_last_run",), "core"),
    FeatureRequirement("馬匹", " handicapping weight", ("handicap_weight",), "extra"),
    FeatureRequirement("馬匹", "產地 / country", ("country",), "extra"),
    FeatureRequirement("馬匹", "性別 / sex", ("sex",), "extra"),
    FeatureRequirement("馬匹", "進口類別 / import type", ("import_type",), "extra"),
    # ---- 往績 -----------------------------------------------------------------
    FeatureRequirement("往績", "近 N 場名次 / recent form", ("form_mean", "form_win_rate"), "core"),
    FeatureRequirement("往績", "分段時間(只用過往賽事) / sectional speed", ("sec_per_metre",), "core"),
    FeatureRequirement("往績", "距離適應度 / distance suitability", ("horse_id_distance_m_prior_rate",), "core"),
    FeatureRequirement("往績", "場地適應度 / going suitability", ("horse_id_going_code_prior_rate",), "core"),
    # ---- 騎練 -----------------------------------------------------------------
    FeatureRequirement("騎練", "騎師累積勝率 / jockey win rate", ("jockey_id_prior_rate",), "core"),
    FeatureRequirement("騎練", "練馬師累積勝率 / trainer win rate", ("trainer_id_prior_rate",), "core"),
    FeatureRequirement("騎練", "騎練組合 / jockey-trainer combo", ("jockey_id_trainer_id_prior_rate",), "core"),
    # ---- 文字 -----------------------------------------------------------------
    FeatureRequirement("文字", "獸醫報告 / veterinary record", (), "unavailable",
                       "兩個 endpoint（veterinaryrecord / overecord）都唔按馬匹過濾，"
                       "每匹馬都係同一份 population 名單"),
    FeatureRequirement("文字", "傷患 / injury (days since)", (), "unavailable",
                       "同上：冇逐馬嘅傷患紀錄"),
    FeatureRequirement("文字", "晨操評語 / trackwork notes", (), "unavailable",
                       "HKJC 只公開當日 trackwork，冇歷史賽事嘅評語"),
    FeatureRequirement("文字", "馬評 / pre-race preview", (), "unavailable",
                       "馬評係有嘅，但只喺每個賽馬日嘅 Form Guide PDF 入面；"
                       "普通賽事冇可推導嘅 PDF URL，只有特殊賽事有存檔"),
    # ---- 市場 -----------------------------------------------------------------
    FeatureRequirement("市場", "賠率 / win odds", ("win_odds",), "core",
                       "只有 SP（開跑後定價），非開盤賠率"),
    FeatureRequirement("市場", "開盤賠率 / opening odds", (), "unavailable",
                       "HKJC 冇歷史開盤賠率"),
    FeatureRequirement("市場", "賠率走勢 / odds movement", (), "unavailable",
                       "HKJC 冇歷史賠率走勢；呢個係最關鍵嘅缺口"),
)


def audit_features(frame: pd.DataFrame, feature_set=None) -> pd.DataFrame:
    """Check each specified feature against the collected data.

    Two questions per feature:

    * **implemented** — does the feature builder actually produce it?  Many
      of the features are *derived* (a jockey's pre-race win rate, the horse's
      seconds-per-metre) and do not exist as a column in the raw file, so the
      audit looks at the built feature set as well.
    * **coverage** — what fraction of the rows actually carry the underlying
      data?
    """
    built: set[str] = set()
    built_frame = None
    if feature_set is not None:
        built = set(feature_set.numeric_features) | set(feature_set.categorical_features)
        built_frame = feature_set.frame

    rows = []
    for req in FEATURE_SPEC:
        present = [c for c in req.columns if c in frame.columns or c in built]
        coverage = 0.0
        for col in req.columns:
            # A stored column reports its own availability; a derived feature
            # reports it from the built feature table.
            source = frame if col in frame.columns else built_frame
            if source is not None and col in source.columns:
                share = float(source[col].notna().mean())
                coverage = max(coverage, share)

        derived = [c for c in req.columns if c in built and c not in frame.columns]
        implemented = bool(present)

        if req.status == "unavailable" or not implemented:
            state = "unavailable"
        elif coverage >= 0.5:
            state = "covered"
        elif coverage > 0:
            state = "partial"
        else:
            state = "derived-only" if derived else "empty"

        rows.append(
            {
                "category": req.category,
                "feature": req.name,
                "state": state,
                "coverage": coverage,
                "columns": ", ".join(present) if present else "—",
                "note": req.note,
            }
        )
    return pd.DataFrame(rows)


def coverage_table(frame: pd.DataFrame, feature_set=None) -> str:
    """A readable coverage report."""
    table = audit_features(frame, feature_set)
    order = {"covered": 0, "partial": 1, "derived-only": 2, "empty": 3, "unavailable": 4}
    table = table.sort_values(
        ["category", "state"], key=lambda s: s.map(order) if s.name == "state" else s
    )
    labels = {
        "covered": "OK      ",
        "partial": "PARTIAL ",
        "derived-only": "DERIVED ",
        "empty": "EMPTY   ",
        "unavailable": "N/A     ",
    }
    lines = []
    current = None
    for _, r in table.iterrows():
        if r["category"] != current:
            current = r["category"]
            lines.append("")
            lines.append(f"── {current} " + "─" * (60 - len(current)))
        note = f"  ({r['note']})" if r["note"] else ""
        lines.append(
            f"   {labels[r['state']]} {r['feature']:<42s} {r['coverage']:6.1%}  {r['columns']}{note}"
        )
    counts = table["state"].value_counts().to_dict()
    lines.append("")
    lines.append(
        "summary: "
        + ", ".join(f"{k}={v}" for k, v in counts.items())
    )
    return "\n".join(lines)
