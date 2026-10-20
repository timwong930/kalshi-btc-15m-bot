# Read-only virtual trade path audit (Phase 5d)

## Purpose

The Phase 5c entry-stop audit found **zero** entries where the *unchanged* entry bid alone crossed the modeled -$0.10 net stop. This follow-up measures what happened between recorded entry and exit **quotes** for each CLOSED virtual trade.

It does not place orders, change services, replay alternative stop policies, or fabricate historical fills.

## Reproducible Mac Studio validation

Use a new isolated worktree for branch `feature/paper-trade-path-audit` based on draft PR #7. **Do not** overwrite or switch worktrees backing the four tmux processes.

```bash
cd ~/Documents/kalshi-btc15m-trade-path
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m src.paper_trade_path_audit \
  --research-db /Users/tim/Documents/kalshi-btc-15m-research/data/research_stream.db \
  --paper-db /Users/tim/Documents/kalshi-btc15m-paper/data/paper_research/paper.db \
  --include-trades
```

Both SQLite databases use `mode=ro` and `PRAGMA query_only=ON`. No schema changes, raw DB uploads, network calls, credentials, or order paths.

## Metrics and interpretation

For each strategy, the JSON report provides:

- CLOSED / CENSORED / OPEN counts. Only CLOSED virtual trades get path reconstruction. Censored positions are **not** assigned favorable or unfavorable P&L.
- Source integrity: exact recorded entry/exit quote IDs, ticker, source, timestamps, side prices and P&L must match. Inconsistent records are counted in `unmatched_reasons`.
- `entry_spread`: recorded ask minus bid at entry.
- `exit_bid_change_from_entry_bid`: subsequent change in bid, **not** a hypothetical midpoint fill.
- `gross_quote_movement`: exit bid minus entry ask, exactly equal to `exit_bid_change_from_entry_bid - entry_spread`.
- `recorded_round_trip_cost`: recorded entry and exit assumed fees and slippage.
- `recorded_virtual_net_pnl`: gross quote movement minus modeled costs, checked for reconciliation.
- Elapsed seconds and number of same-ticker/source quotes between entry and exit.
- Whether the **first later structurally valid quote**, at least one second after entry, already meets the -$0.10 modeled net stop. This is not a replay of processing-time freshness and cannot prove a fill occurred.
- Stops occurring within 15, 30, or 60 seconds.
- Optional Coinbase as-of price change, limited to observed prices received within the prior 10 seconds, signed for UP/DOWN. **Coinbase is not Kalshi's BRTI settlement index**. Missing and stale BTC observations remain missing.

This is a descriptive analysis of already-closed quote-based virtual trades, not actual fills or proof of an independently tested winning strategy. Do not retune policies against this small sample and call it out-of-sample validation.

## Hermes — validation only

1. Fetch the PR branch into **a new isolated worktree**; do not modify existing repositories or processes.
2. Run complete `pytest -q` and `ruff check .`. If either fails, return output; **do not edit files**.
3. On PASS, execute the read-only command above with `--include-trades`. Return full JSON or a faithful summary and locally retained report. Do not upload private DBs or secrets.
4. Leave all four tmux services running. No model promotion, PR merge, database writes, deployment, credentials, exchange orders, or trading activation.
