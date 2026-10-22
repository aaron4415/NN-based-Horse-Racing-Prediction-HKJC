"""Diagnostic figures.

The report asks for two pictures that a numbers-only report cannot replace:

* a **reliability curve** — "does a horse the model gives 20% actually win
  20% of the time?" — with the market on the same axes, because a model is
  only interesting relative to the market;
* the **equity curve** of the betting simulation, because a single ROI number
  hides whether the profit was made steadily or in one lucky week.

Matplotlib is imported lazily so that the rest of the package works without
it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required for the diagnostic figures: pip install matplotlib"
        ) from exc
    return plt


def _percent_formatter(decimals: int = 0):
    """Axis formatter that renders a 0..1 fraction as a percentage."""
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _pos: f"{v * 100:.{decimals}f}%")


def plot_calibration(
    tables: dict[str, pd.DataFrame],
    path: str | Path,
    *,
    title: str = "Reliability curve (test split)",
    diagonal_label: str = "perfect calibration",
    min_count: int = 25,
) -> Path:
    """Plot reliability curves for several models against the market.

    Parameters
    ----------
    tables
        Mapping of label -> reliability table (the output of
        :func:`hkjc_nn.evaluation.calibration_table`).
    path
        Where to write the PNG.
    min_count
        Bins holding fewer runners than this are dropped.  A reliability
        diagram computed from a handful of runners is noise, and drawing it
        turns the figure into a zig-zag that says nothing.
    """
    plt = _pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for label, table in tables.items():
        if table is None or table.empty:
            continue
        x = table["mean_predicted"].to_numpy(dtype=float)
        y = table["observed_frequency"].to_numpy(dtype=float)
        n = table["n"].to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y) & (n >= min_count)
        if not keep.any():
            continue
        # Marker size tracks the bin's sample count.
        biggest = n[keep].max()
        sizes = 26.0 + 90.0 * np.sqrt(n[keep] / max(biggest, 1.0))
        is_market = label.lower().startswith("market")
        colour = "black" if is_market else None
        ax.plot(
            x[keep],
            y[keep],
            lw=1.6,
            marker="o",
            ms=4,
            alpha=0.85,
            color=colour,
            ls="--" if is_market else "-",
            label=label,
        )
        ax.scatter(
            x[keep],
            y[keep],
            s=sizes,
            zorder=3,
            alpha=0.85,
            color=colour,
            marker="s" if is_market else "o",
        )

    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls=":", label=diagonal_label)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed win frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    ax.text(
        0.98,
        0.02,
        f"marker size ∝ bin count (bins with n < {min_count} dropped)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="grey",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_model_comparison(
    metrics: "pd.DataFrame",
    path: str | Path,
    *,
    split: str = "test",
    title: str = "Race log-loss: every model against the market",
) -> Path:
    """Bar chart of each model's race log-loss with the market as a reference line.

    Lower is better.  The market is drawn as a vertical line, so the reader
    sees at a glance which models clear it and which do not — which is the
    entire finding of the project.
    """
    plt = _pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = metrics[metrics["split"] == split].copy()
    if rows.empty:
        raise ValueError(f"no rows for split {split!r}")

    market = rows["vs_market_ll_market"].iloc[0]
    ordered = rows.sort_values("race_log_loss", ascending=False)

    labels = {
        "A": "A  conditional logit",
        "B": "B  LightGBM",
        "C": "C  listwise MLP",
        "D": "D  MLP + text",
        "E": "E  LightGBM + text",
        "F": "F  blend + market",
    }

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ys = range(len(ordered))
    label_x = ordered["race_log_loss"].max() + 0.012
    for y, (_, r) in zip(ys, ordered.iterrows()):
        code = r["model"]
        better = r["race_log_loss"] < market
        colour = "#2f6f5e" if better else "#b06367"
        ax.barh(y, r["race_log_loss"], color=colour, alpha=.88, height=.62, zorder=3)
        ax.text(
            label_x,
            y,
            f"{r['race_log_loss']:.4f}    R² {r['vs_market_r2']:+.4f}",
            va="center",
            fontsize=10,
            color="#3a4256",
            fontfamily="monospace",
        )

    ax.axvline(market, color="#14171e", ls="--", lw=1.4, zorder=4)

    ax.set_yticks(list(ys))
    ax.set_yticklabels([labels.get(m, m) for m in ordered["model"]], fontsize=10.5)
    ax.set_xlabel(
        f"race log-loss  (lower is better)   ·   dashed line = the market, {market:.4f}"
    )
    ax.set_xlim(min(market, ordered["race_log_loss"].min()) - 0.02, label_x + 0.13)
    ax.set_ylim(-0.7, len(ordered) - 0.3)
    ax.set_title(title, fontsize=13, color="#14171e", pad=14)
    ax.grid(axis="x", color=(0, 0, 0, 0.06), zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#e9e3d6")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_equity(equity: pd.DataFrame, path: str | Path, *, title: str = "Bankroll") -> Path:
    """Plot the bankroll over time from the betting simulation.

    The y-axis switches to a log scale when the bankroll spans more than one
    order of magnitude.  Fractional Kelly compounds hard, and on a linear axis
    the first months — where the strategy is actually being tested — are
    squashed flat against the axis.
    """
    plt = _pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    if equity is not None and not equity.empty:
        values = equity["bankroll"].to_numpy(dtype=float)
        ax.plot(
            pd.to_datetime(equity["meeting_date"]),
            values,
            color="#2b6cb0",
            lw=1.4,
        )
        positive = values[values > 0]
        if len(positive) and positive.max() / max(positive.min(), 1e-9) > 50:
            ax.set_yscale("log")
            ax.set_ylabel("bankroll (log scale)")
        ax.axhline(
            equity["bankroll"].iloc[0],
            color="grey",
            ls=":",
            lw=1,
            label="starting bankroll",
        )
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("date")
    ax.set_ylabel("bankroll")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_yearly_roi(yearly: pd.DataFrame, path: str | Path, *, title: str = "ROI by year") -> Path:
    """Plot the year-on-year ROI of the betting simulation.

    Accepts either the aggregated table from
    :func:`hkjc_nn.betting.yearly_roi` or the raw bets frame.
    """
    plt = _pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = yearly.copy()
    if "roi" not in df.columns:
        if {"year", "profit", "staked"} <= set(df.columns):
            df["roi"] = df["profit"] / df["staked"].replace(0, float("nan"))
        else:
            df["year"] = pd.to_datetime(df["meeting_date"]).dt.year
            grouped = df.groupby("year").agg(staked=("stake", "sum"), profit=("profit", "sum"))
            df = grouped.reset_index()
            df["roi"] = df["profit"] / df["staked"].replace(0, float("nan"))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if not df.empty:
        colours = ["#2f6f5e" if v >= 0 else "#b06367" for v in df["roi"]]
        ax.bar(df["year"].astype(str), df["roi"], color=colours)
        ax.axhline(0.0, color="#14171e", lw=1)
        for x, v in zip(df["year"].astype(str), df["roi"]):
            ax.annotate(
                f"{v:+.0%}",
                (x, v),
                textcoords="offset points",
                xytext=(0, 5 if v >= 0 else -15),
                ha="center",
                fontsize=10,
                fontfamily="monospace",
                color="#3a4256",
            )
        # Leave room for the value labels so they never sit on the axis.
        lo, hi = float(df["roi"].min()), float(df["roi"].max())
        span = max(hi - lo, 0.05)
        ax.set_ylim(min(lo - 0.16 * span, -0.02), max(hi + 0.16 * span, 0.02))
    ax.set_xlabel("year")
    ax.set_ylabel("ROI")
    ax.yaxis.set_major_formatter(_percent_formatter())
    ax.set_title(title, fontsize=13, color="#14171e", pad=12)
    ax.grid(axis="y", color=(0, 0, 0, 0.05), zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#e9e3d6")
    ax.spines["bottom"].set_color("#e9e3d6")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
