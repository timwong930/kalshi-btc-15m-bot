# Phase 4 — Prospective, quotes-only virtual paper trading

**What this is:** a self-contained local simulation of future hypothetical
BUY UP / BUY DOWN entries and later quote-based exits for six predeclared
strategies. Reads the existing Kalshi research SQLite DB in `mode=ro` and
writes a **separate local paper ledger**. No exchange orders of any kind,
including Kalshi demo orders. No API calls, credentials, account access,
automatic model selection, or model promotion.

This is **prospective forward paper research**, not a retrospective backtest
of historical quotes. On the first launch, the ledger saves the collector's
latest quote ID and **does not trade any previously collected quotes**.
On restart, it **censors still-open simulated positions and fast-forwards
past all unobserved quotes**. It will **not simulate trades or exits at prices
from downtime**. Restarting is a conservative reset, not a replay.

### Strategies running head-to-head (one simulated contract, max one entry per
market per strategy)

- `price_rule`: buy the lower-priced side if its ask is 15–35 cents.
- `momentum_rule`: follow a sufficiently large Coinbase 60-second
  price move, provided the corresponding contract ask is ≤75 cents.
- `outcome_logistic` and `outcome_boosted_stumps`: compare the **frozen**
  shadow outcome probability to the actual ask plus assumed costs; take
  only an edge of ≥8 cents.
- `reversal_logistic` and `reversal_boosted_stumps`: enter only if the
  frozen research model estimates ≥70% chance of hitting the take-profit
  before the stop, with a contract ask ≤80 cents.
- `no_trade` baseline always earns $0, with zero modeled risk.

The ML rules only use artifacts whose file modification timestamp **and**
last training-market close precede the potential trade timestamp. If a
model is missing, incompatible, or ineligible, the corresponding strategy
records SKIP. The simulator never automatically changes training or
fitting, uses no settlement result for entries, and never reads the
old settled-outcomes DB. These strategies are **experimental**, not
individually validated profit-generating signals.

### Entry/exit policy

- First eligible current quote **between T-5 and T-4 minutes** per market.
  Explicit source, quote ID, timestamp, entry ask and opposite-book depth
  are saved. Only complete spread/depth quotes (≥5 contracts, ≤8¢ spread),
  with ≤25s ingestion freshness, can open a paper entry.
- All strategies pay a hypothetical **$0.02 fee and $0.01 slippage per
  contract per side**. These are sensitivity assumptions, **not a
  confirmed Kalshi fee schedule**.
- Exit at observed bid on the FIRST observed +10¢ net take-profit, −10¢
  net stop-loss, or within 45s before scheduled close. Wait at least one
  second after entry. No fills at midprice, no delayed/book-lookahead exits.
- If quotes are delayed by >25s, consecutive exit quotes have a gap >30s,
  market book/depth becomes invalid, or no executable quote exists near
  expiry, mark **CENSORED**, with no P&L. This avoids fabricating exits.
  Reports separately show censor counts: P&L on closed trades can be
  upward-biased when censoring is common. Never pretend these are
  exchange-confirmed fills.
- Results are *one-contract, per-strategy counterfactuals*. Strategies
  may enter simultaneously, so summed across strategies is not an
  independently risk-managed portfolio. Executability and queue position
  have **not** been validated.

### Output and operational checks

- `data/paper_research/paper.db`: SQLite with `paper_meta`,
  `paper_decisions`, `paper_trades`. Entries, SKIPs, model hashes,
  evidence IDs, exits and censor reasons are persisted and idempotent.
- `data/paper_research/latest_metrics.json`: simulated P&L per
  one-contract trade, completed count, win rate, gross wins/losses,
  expectancy, profit factor, closed-trade drawdown, pending/censored.
- `data/paper_research/paper_trades.csv`: per-trade audit.
- `data/paper_research/equity_curves.csv`: per-strategy simulated
  closed-trade equity and drawdown.
- `logs/paper_sim.log`: runtime log.
- Exactly one simulator per paper-ledger folder, enforced with an
  exclusive file lock. Existing tmux services are never modified.

## Hermes deployment on Mac Studio

**Do not merge PR #4.** Keep the three current `kalshi-btc` windows
running untouched. Work on a fourth isolated worktree:

```bash
cd ~/Documents/kalshi-btc-15m-bot
git fetch origin
git worktree add -b feature/prospective-paper-simulator \
  ~/Documents/kalshi-btc15m-paper origin/feature/prospective-paper-simulator
cd ~/Documents/kalshi-btc15m-paper
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

If that branch/worktree already exists locally, inspect and reuse it
rather than creating another.

**Verify all tests pass before starting a process.** First run once:

```bash
cd ~/Documents/kalshi-btc15m-paper
.venv/bin/python -m src.paper_sim \
  --research-db /Users/tim/Documents/kalshi-btc-15m-research/data/research_stream.db \
  --paper-db data/paper_research/paper.db \
  --model-dir /Users/tim/Documents/kalshi-btc15m-ml/data/ml_research/shadow_models \
  --reports-dir data/paper_research
```

The initial status will show no historical trades: **by design**.
Then launch the live, local paper simulation in a **fourth** tmux window
in the **existing** `kalshi-btc` session:

```bash
tmux new-window -d -t kalshi-btc -n paper-sim
tmux send-keys -t kalshi-btc:paper-sim \
  'cd ~/Documents/kalshi-btc15m-paper && mkdir -p logs && .venv/bin/python -m src.paper_sim --loop --research-db /Users/tim/Documents/kalshi-btc-15m-research/data/research_stream.db --paper-db data/paper_research/paper.db --model-dir /Users/tim/Documents/kalshi-btc15m-ml/data/ml_research/shadow_models --reports-dir data/paper_research 2>&1 | tee -a logs/paper_sim.log' C-m
```

Inspect at any time:

```bash
cd ~/Documents/kalshi-btc15m-paper
.venv/bin/python -m src.paper_sim --status \
  --paper-db data/paper_research/paper.db \
  --reports-dir data/paper_research
tmux list-windows -t kalshi-btc
tail -n 50 logs/paper_sim.log
```

Reports refresh at least every five minutes. First eligible future market
arrivals may take a while; zero trades on first launch is a correct
state. The simulator only starts recording *new* eligible entries after
initialization.

### Acceptance criteria

1. All four windows alive: `Python`, `research-collector`,
   `ml-trainer`, `paper-sim`; no restarts of the first three.
2. All tests and Ruff pass, including entry/exit event ordering, fees,
   stop/take-profit, gap censoring, no hindsight model artifacts, idempotent
   cursor, never replaying old records, independent ledger and no-trade.
3. Observe new quote cursor progress in the fourth window; eventually
   observe decisions (even if only SKIPs) for future markets.
4. Original research and predictor databases are not modified by
   `paper-sim`; existing keys and trading flags remain untouched.
5. Confirm **zero Kalshi demo/production orders** and zero use of
   API credentials by the new simulator.
6. Keep the new PR **draft**. Commit only legitimate test fixes on its
   own feature branch, never restart/modify live collector or predictor.

### Interpretation

Early trade counts will be small. A positive simulated P&L is not
evidence of statistical edge until **many unseen markets and market
regimes** have been tested. Censoring and selection effects can bias
the results. These paper returns assume fills at observed quotes,
slippage fixed in advance, and a flat per-side fee. Before ever using
real money, verify the applicable actual Kalshi fee schedule and
order-book execution likelihood, then evaluate drawdown on an
independent holdout dataset. **Nothing here is approved to trade.**
