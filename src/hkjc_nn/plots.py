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
    """Plot the year-on-year ROI of the betting simulation."""
    plt = _pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if yearly is not None and not yearly.empty:
        colours = ["#2b6cb0" if v >= 0 else "#c53030" for v in yearly["roi"]]
        ax.bar(yearly["year"].astype(str), yearly["roi"], color=colours)
        ax.axhline(0.0, color="black", lw=1)
    ax.set_xlabel("year")
    ax.set_ylabel("ROI")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
