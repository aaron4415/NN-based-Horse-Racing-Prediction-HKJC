"""Command line interface.

Examples
--------
Generate a synthetic dataset and write it out for inspection::

    hkjc-nn make-data --races 3000 --out data/synthetic.csv

Run the full experiment matrix on that data::

    hkjc-nn run --data data/synthetic.csv --train-until 2022-12-31 --val-until 2023-12-31

Run it end to end on simulated data (no files needed)::

    hkjc-nn run --synthetic --races 2500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _cmd_make_data(args: argparse.Namespace) -> int:
    from .data import simulate_races, write_canonical_csv

    df = simulate_races(
        n_races=args.races,
        n_horses=args.horses,
        seed=args.seed,
        market_skill=args.market_skill,
        market_specialist=args.market_specialist,
        market_condition=args.market_condition,
        market_noise=args.market_noise,
        signal_noise=args.signal_noise,
        takeout=args.takeout,
    )
    out = Path(args.out)
    write_canonical_csv(df, out)
    print(f"wrote {len(df):,} runner rows across {df['race_id'].nunique():,} races -> {out}")

    from .schema import validate_frame

    print(validate_frame(df).summary())
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .betting import BetConfig
    from .experiments import ExperimentConfig, run_experiments
    from .features import FeatureConfig

    betting = BetConfig(
        kelly_fraction=args.kelly_fraction,
        takeout=args.takeout,
        payout_model=args.payout_model,
        max_bet_pct=args.max_bet_pct,
        max_race_pct=args.max_race_pct,
        start_bankroll=args.bankroll,
        odds_haircut=args.odds_haircut,
        label=args.label,
    )

    cfg = ExperimentConfig(
        data_source="synthetic" if args.synthetic else args.data,
        synthetic_kwargs={
            "n_races": args.races,
            "n_horses": args.horses,
            "seed": args.seed,
            "market_skill": args.market_skill,
            "market_specialist": args.market_specialist,
            "market_condition": args.market_condition,
            "market_noise": args.market_noise,
            "signal_noise": args.signal_noise,
        },
        train_until=args.train_until,
        val_until=args.val_until,
        feature_config=FeatureConfig(),
        text_dim=args.text_dim,
        use_text=not args.no_text,
        models=tuple(args.models),
        nn_architecture=args.nn_architecture,
        nn_epochs=args.nn_epochs,
        seed=args.seed,
        betting=betting,
        devig_method=args.devig,
    )

    results = run_experiments(cfg)
    print(results.summary())

    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        results.metrics.to_csv(args.metrics_out, index=False, encoding="utf-8-sig")
        print(f"\nmetrics -> {args.metrics_out}")
    if args.predictions_out:
        Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
        results.predictions.to_csv(args.predictions_out, index=False, encoding="utf-8-sig")
        print(f"predictions -> {args.predictions_out}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics": results.metrics.to_dict(orient="records"),
            "betting": results.betting_summary,
            "blend": results.blend_summary,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"summary json -> {args.json_out}")
    if args.plot_dir:
        from .plots import plot_calibration, plot_equity, plot_yearly_roi

        out = Path(args.plot_dir)
        out.mkdir(parents=True, exist_ok=True)
        written = [
            plot_calibration(results.calibration, out / "calibration.png"),
        ]
        if not results.equity_curve.empty:
            written.append(plot_equity(results.equity_curve, out / "equity.png"))
        if not results.yearly.empty:
            written.append(plot_yearly_roi(results.yearly, out / "yearly_roi.png"))
        for path in written:
            print(f"figure -> {path}")
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .collect.hkjc import CollectConfig, collect_races, enrich_horses

    cfg = CollectConfig(
        start=args.start,
        end=args.end,
        target_races=args.target_races,
        cache_dir=Path(args.cache_dir),
        out_path=Path(args.out),
        horses_path=Path(args.horses_out),
        delay=args.delay,
        workers=args.workers,
        enrich_horses=not args.no_horses,
        enrich_veterinary=not args.no_veterinary,
    )

    frame = collect_races(cfg)
    if frame.empty:
        print("nothing collected")
        return 1

    if not args.no_horses:
        frame = enrich_horses(frame, cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    print(
        f"\nwrote {len(frame):,} runner rows across {frame['race_id'].nunique():,} races -> {out}"
    )
    if "rating" in frame.columns:
        print(f"ratings present for {frame['rating'].notna().mean():.1%} of rows")
    return 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .coverage import audit_features, coverage_table
    from .data.csv_adapter import load_csv_races
    from .features import build_features

    frame = load_csv_races(Path(args.data))
    # Many of the specified features are derived rather than stored, so the
    # audit is run against the built feature set as well as the raw columns.
    feature_set = build_features(frame)
    print(coverage_table(frame, feature_set))
    if args.out:
        audit_features(frame, feature_set).to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\ncoverage table -> {args.out}")
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    from .data.csv_adapter import describe_csv

    table = describe_csv(args.path)
    print(table.to_string(index=False))
    missing = table.loc[~table["present"], "canonical"].tolist()
    if missing:
        print(f"\ncolumns the file cannot populate: {', '.join(missing)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hkjc-nn", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # -- make-data ------------------------------------------------------------------
    p = sub.add_parser("make-data", help="generate a synthetic dataset")
    p.add_argument("--races", type=int, default=3000)
    p.add_argument("--horses", type=int, default=260)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--market-skill", type=float, default=1.0,
                   help="how much of the visible information the market prices in")
    p.add_argument("--market-specialist", type=float, default=0.80,
                   help="how much of the distance/going specialist layer the market prices in")
    p.add_argument("--market-condition", type=float, default=0.0,
                   help="how much of the day-time fitness layer the market prices in")
    p.add_argument("--market-noise", type=float, default=0.55,
                   help="the market's own information noise; 0 = an unbeatable market")
    p.add_argument("--signal-noise", type=float, default=0.35)
    p.add_argument("--takeout", type=float, default=0.225)
    p.add_argument("--out", default="data/synthetic.csv")
    p.set_defaults(func=_cmd_make_data)

    # -- run ------------------------------------------------------------------------
    p = sub.add_parser("run", help="run the experiment matrix")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="use simulated data")
    source.add_argument("--data", help="path to a canonical CSV extract")
    p.add_argument("--races", type=int, default=2500)
    p.add_argument("--horses", type=int, default=220)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--market-skill", type=float, default=1.0)
    p.add_argument("--market-specialist", type=float, default=0.80)
    p.add_argument("--market-condition", type=float, default=0.0)
    p.add_argument("--market-noise", type=float, default=0.55)
    p.add_argument("--signal-noise", type=float, default=0.35)
    p.add_argument("--train-until", default="2022-12-31")
    p.add_argument("--val-until", default="2023-12-31")
    p.add_argument("--models", nargs="+", default=["A", "B", "C", "D", "E", "F"])
    p.add_argument("--nn-architecture", default="field_transformer",
                   choices=["field_transformer", "listwise_mlp"])
    p.add_argument("--nn-epochs", type=int, default=40)
    p.add_argument("--text-dim", type=int, default=12)
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--devig", default="basic", choices=["shin", "basic"],
                   help="how to remove the margin from the odds; 'basic' is correct for a pari-mutuel pool")
    p.add_argument("--kelly-fraction", type=float, default=0.25)
    p.add_argument("--takeout", type=float, default=0.225)
    p.add_argument("--odds-haircut", type=float, default=0.0,
                   help="haircut applied to the published price (HKJC prices are starting prices)")
    p.add_argument("--payout-model", default="quoted", choices=["quoted", "pari_mutuel", "effective"])
    p.add_argument("--max-bet-pct", type=float, default=0.02)
    p.add_argument("--max-race-pct", type=float, default=0.05)
    p.add_argument("--bankroll", type=float, default=10_000.0)
    p.add_argument("--label", default="default")
    p.add_argument("--metrics-out")
    p.add_argument("--predictions-out")
    p.add_argument("--json-out")
    p.add_argument("--plot-dir", help="write calibration / equity / yearly-ROI figures here")
    p.set_defaults(func=_cmd_run)

    # -- describe -------------------------------------------------------------------
    p = sub.add_parser("describe", help="show which canonical columns a CSV can populate")
    p.add_argument("path")
    p.set_defaults(func=_cmd_describe)

    # -- collect --------------------------------------------------------------------
    p = sub.add_parser("collect", help="collect real HKJC race results")
    p.add_argument("--start", default="2018-01-01", help="first meeting date to consider")
    p.add_argument("--end", default="2025-12-31", help="last meeting date to consider")
    p.add_argument("--target-races", type=int, default=5000)
    p.add_argument("--out", default="data/hkjc.csv")
    p.add_argument("--horses-out", default="data/hkjc_horses.csv")
    p.add_argument("--cache-dir", default="data/hkjc_cache")
    p.add_argument("--delay", type=float, default=0.35,
                   help="minimum seconds between requests (global, not per-thread)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-horses", action="store_true",
                   help="skip the horse-page pass (no ratings)")
    p.add_argument("--no-veterinary", action="store_true",
                   help="skip the veterinary-record pass (no injury notes)")
    p.set_defaults(func=_cmd_collect)

    # -- coverage ------------------------------------------------------------------
    p = sub.add_parser("coverage", help="audit feature coverage against the spec")
    p.add_argument("--data", default="data/hkjc_races.csv")
    p.add_argument("--out")
    p.set_defaults(func=_cmd_coverage)

    return parser


def main(argv: list[str] | None = None) -> int:
    # The Windows console defaults to a legacy code page (cp950/cp437) that
    # cannot render the Chinese feature names, so force UTF-8 output.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - non-tty
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
