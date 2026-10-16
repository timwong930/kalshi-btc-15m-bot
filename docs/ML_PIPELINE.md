# Phase 2 + 3: autonomous, read-only ML research

This is **not** an automatic trading model. It is a third, independent
process that watches the two existing SQLite databases, validates historical
observations, builds labels, and trains two research-only challengers after
enough **distinct settled** 15-minute BTC markets have been collected.

No new Kalshi API requests are made, no production credentials are loaded,
and no changes are made to the predictor, collector, or existing trading
settings. This must remain a **draft PR** until smoke-tested on the Mac Studio.

## Phase 2 — quality and labeling

On each **newly settled research market** (checked every 15 minutes):

- Reads `research_stream.db` from the collector **read-only** and
  `kalshi_btc15m.db` settlement outcomes from the existing predictor
  **read-only**; never mutates either source.
- Chooses **one** eligible quote closest to five minutes before market
  close, within a predeclared 4–7 minute range, for *outcome prediction*.
  Uses only BTC/BRTI ticks received **at or before** the quote timestamp;
  skips stale prices, invalid strike, missing short-volatility history,
  missing quotes/depth, and unfinalized outcomes.
- Writes `outcome_dataset.csv` with the historical market probability,
  spot vs strike, volatility, spread, book depth and time remaining, plus
  the *settled result*, but no future outcomes as model features.
- Regenerates `reversal_labels.csv` from the existing fee/slippage-aware
  replay, retaining `incomplete_quotes` instead of treating coverage gaps
  as a failed reversal. Creates a separate `reversal_dataset.csv` using
  only complete, settled, predeclared T-5m observations, one per side per
  market. Train/test splits group both sides of the same market together.
- Publishes `latest_report.json` with settled vs usable market counts,
  exclusions, incomplete reversal history, and stream totals.

Old closed-market records are **valuable** and never automatically deleted.
Historical data recorded before the enhanced collector began is not
fabricated into new features.

## Phase 3 — shadow ML training and evaluation

After **at least 80 distinct usable settled markets** (defaults: 60
training markets + 20 test markets), evaluates both:

1. Regularized logistic regression.
2. Conservative gradient-boosted decision stumps (small tree ensemble).

Both implementations use Python's standard library: no heavyweight
frameworks, binary wheels, or external services required.

**Walk-forward evaluation** fits models only to markets that settled
*before* each test market's entry, in expanding time-ordered folds. All
quotes and both sides of a given market stay in the same cohort.
Scaling, baseline event prevalence, and split thresholds are
learned using training data only.

- Outcome baseline: the observed Kalshi YES bid/ask midpoint.
- Reversal baseline: the take-profit event rate in preceding TRAIN markets.
- Also reports **hypothetical one-contract buy-and-hold settlement returns**
  for test-market candidates, using observed ask plus assumed buying fees and
  slippage (never actual fills or realistic queue execution).
- Reports Brier score and log-loss for challenger and baseline, paired Brier
  differences by **distinct market**, fold details, and a descriptive
  normal-approximation uncertainty interval. Evidence remains exploratory
  because nearby markets may be correlated.
- Generates a new, versioned JSON **shadow** model for offline inspection.
  Final artifacts trained on all labelled data are not used to report
  out-of-sample performance; that comes solely from the walk-forward folds.
- Does **not** automatically select a winning model, update predictor code,
  change edge thresholds, execute paper/live orders, or change position size.
  Better-looking Brier scores alone cannot establish trading profitability.

The loop checks every 15 minutes and runs expensive model evaluation
**at most daily** by default. New settlement labels are prepared on
intermediate cycles. The report says `WAITING_FOR_MORE_SETTLED_MARKETS`
until enough usable data exists, or `TRAINING_COOLDOWN` between full
training runs. Models remain shadow-only even if statistically better.

## Recommended Mac Studio setup (new worktree)

Hermes should preserve the existing, healthy `kalshi-btc` tmux windows:
`Python` (existing `--predict`) and `research-collector`.
Do not restart those processes or merge PR #1/#2.

From the existing repo (adjust paths only if actual layout differs):

```bash
cd ~/Documents/kalshi-btc-15m-bot
git fetch origin
git worktree add ~/Documents/kalshi-btc15m-ml origin/feature/validated-ml-challengers
cd ~/Documents/kalshi-btc15m-ml
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

A worktree created directly from `origin/<branch>` can start detached;
Hermes can attach a matching local tracking branch if desired without
switching either running worktree.

Run **once first**, using ABSOLUTE source paths (DO NOT point the
trainer at a local empty SQLite file):

```bash
cd ~/Documents/kalshi-btc15m-ml
.venv/bin/python -m src.ml_pipeline \
  --research-db /Users/tim/Documents/kalshi-btc-15m-research/data/research_stream.db \
  --outcomes-db /Users/tim/Documents/kalshi-btc-15m-bot/data/kalshi_btc15m.db \
  --output-dir data/ml_research
.venv/bin/python -m src.ml_pipeline --status --output-dir data/ml_research
```

When the one-shot pass and tests succeed, run in a *new window* in the
**existing** `kalshi-btc` session:

```bash
tmux new-window -d -t kalshi-btc -n ml-trainer
tmux send-keys -t kalshi-btc:ml-trainer \
  'cd ~/Documents/kalshi-btc15m-ml && mkdir -p logs && .venv/bin/python -m src.ml_pipeline --loop --research-db /Users/tim/Documents/kalshi-btc-15m-research/data/research_stream.db --outcomes-db /Users/tim/Documents/kalshi-btc-15m-bot/data/kalshi_btc15m.db --output-dir data/ml_research 2>&1 | tee -a logs/ml_trainer.log' C-m
```

Or let Hermes construct an equivalent tmux command to ensure quoting and
logging are correct. **Only one trainer** should run per output directory;
an exclusive file lock rejects duplicate instances.

Inspect:

```bash
tmux list-windows -t kalshi-btc
cd ~/Documents/kalshi-btc15m-ml
.venv/bin/python -m src.ml_pipeline --status --output-dir data/ml_research
tail -n 50 logs/ml_trainer.log
cat data/ml_research/latest_report.json
```

Output files (never upload to a public GitHub repo):

- `data/ml_research/outcome_dataset.csv`
- `data/ml_research/reversal_labels.csv`
- `data/ml_research/reversal_dataset.csv`
- `data/ml_research/latest_report.json`
- `data/ml_research/ml_runs.db` (local run history)
- `data/ml_research/shadow_models/*.json` (immutable offline artifacts)
- `logs/ml_trainer.log`

SQLite collectors should continue collecting while the separate trainer
reads consistent snapshots in WAL mode. Source databases are never
modified; take regular SQLite online backups of the collector and predictor.

## Limitations and future work

- **No ML execution is connected to live monitoring**. Shadow artifacts
  are not loaded by any execution path.
- Sampling one T-5m observation per market improves methodological safety
  but does not provide ML at every time-to-expiry. A later, separately
  validated model can learn multiple horizons using group-aware splits.
- Coinbase REST/WS BTC prices are **not** the BRTI settlement index;
  BRTI as-of observations are used when available. Kalshi REST market
  quotes are asynchronous; paper trade fills, fees and slippage remain
  assumptions, not verified execution.
- The reversal label generator currently replays available history whenever
  new markets settle; monitor CPU/disk as history grows. It is reasonable
  to optimize replay incrementally before storing months of ticks.
- Fewer than 80 *usable markets* means waiting, not fake model accuracy.
  An estimated performance edge is not guaranteed real profit.
