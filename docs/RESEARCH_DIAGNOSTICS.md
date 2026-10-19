# Read-only quote continuity and virtual-paper diagnostics

Run this optional standalone diagnostic without restarting the collector, trainer, predictor, or paper simulator. It opens the collector and paper ledgers in SQLite read-only mode and prints JSON to stdout. It makes no network calls, produces no orders and does not change any running process.

## Mac Studio command (only after checks)

    cd ~/Documents/kalshi-btc15m-paper
    .venv/bin/python -m src.research_diagnostics \
      --research-db /Users/tim/Documents/kalshi-btc-15m-research/data/research_stream.db \
      --paper-db /Users/tim/Documents/kalshi-btc15m-paper/data/paper_research/paper.db

## Interpretations

- Global quote gaps: successive stored quotes separated by over 30 seconds, split into cross-ticker, same-ticker/different-source, or same-ticker/same-source. Cross-market gaps are NOT proof of a collector outage.
- Per-market/source gaps: time gaps between observations from the same source and ticker. Some can reflect sparse order-book changes or data not being recorded.
- T-5 to T-4-minute quote coverage: for markets whose entry windows ended, count markets with any quote, with a static book-eligible quote, or without quotes. First discovery after window opening is also counted.
- Static book eligibility: finite positive bid/ask values below $1, spreads no greater than $0.08 and depths at least five on both sides. This mirrors the current Phase 4 book filter, but does NOT test as-of processing freshness or prove fillability.
- Virtual-paper reasons: per-strategy ENTRY/SKIP reasons, and CLOSED/OPEN/CENSORED status reasons from the existing ledger.

The diagnostic is descriptive, not a performance improvement or a trading strategy. Quote validity flags do not necessarily imply that the book passes spread/depth checks. Censored and open trades remain excluded from completed-trade P&L.

## Change management

Keep this branch as a draft PR stacked on the Phase 4 branch. Ask Hermes to fetch, run the complete pytest and Ruff suites, and execute only the diagnostic command from a new detached or dedicated worktree. Do not overwrite an existing paper worktree or restart any running tmux window. Do not merge or enable live trading.
