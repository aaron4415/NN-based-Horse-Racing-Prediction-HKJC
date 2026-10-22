# Contributing

This project answers one question — *can a neural network beat Benter's
approach on Hong Kong racing?* — and the answer on public data is **no**.
Contributions are welcome, but they are most valuable when they make the
answer more certain, or widen the evidence.

Read [the research flow](docs/research-flow.html) first; it explains how the
parts fit together and which numbers come from where.

## Getting set up

```powershell
git clone https://github.com/aaron4415/NN-based-Horse-Racing-Prediction-HKJC.git
cd NN-based-Horse-Racing-Prediction-HKJC
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev,torch]"
.\.venv\Scripts\python -m pytest -q
```

The test suite runs in under a minute and needs no network access.

## Things that would genuinely help

These are open, self-contained, and each one is a real limitation of the
current work.

### Good first issues

* **Add a feature to the coverage audit.** Open
  `src/hkjc_nn/coverage.py` and add a `FeatureRequirement` for something the
  report asks for that is not yet listed. The audit prints which features the
  collected data actually covers, so this is a small, self-contained change.
* **Add a text encoder.** `src/hkjc_nn/text.py` has a `TextEncoder`
  interface. Implement one that wraps a real encoder (Qwen, BERT, any
  sentence-transformer) and the text branch will use it unchanged. The
  default is TF-IDF + SVD so the repo has no heavy dependencies.
* **Write a blog post or explainer** about the null test, and why a backtest
  that always shows profit is a broken backtest. This is the single most
  useful thing a reader could take away.
* **Translate the README** into 繁體中文 or 简体中文.

### Bigger pieces

* **Odds movement.** The one feature that would change the conclusion, and
  the one HKJC does not publish historically. If you have a licensed feed or
  have built a snapshotter, this is where it goes.
* **A second market.** The same pipeline pointed at a racebook with less
  professional participation would be a genuinely interesting result.
* **Paper trading.** Stage 5 of the roadmap: live forecasts, no money, one
  season. The framework already produces the forecasts; it needs a clock, a
  runner, and a log.

## Ground rules for changes

* **Keep the leakage guards intact.** They are the reason the results are
  trustworthy. If a change makes a post-race column reachable from a
  feature, that is a bug even if the scores improve.
* **The null test must stay green.** If `test_null_market_produces_no_profit`
  fails, the harness is manufacturing profit and the numbers are worthless.
* **Do not commit collected data.** `data/` is gitignored because it holds
  HKJC's content. The collector and the cache are reproducible from the
  public pages.
* **Report numbers, not adjectives.** Every claim in the README should be
  reproducible with a command in the README.

## Style

The code follows whatever `ruff` is configured to check (`ruff check` and
`ruff format` are the defaults). The narrative docs are plain Markdown — no
notebook output, no screenshots of tables that a command can regenerate.

## Licensing

By contributing you agree that your contribution is licensed under the MIT
licence in `LICENSE`.
