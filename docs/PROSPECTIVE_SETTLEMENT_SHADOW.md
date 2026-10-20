# Phase 6 — Prospective virtual hold-to-settlement shadow (NO orders)

## Scope

This is a **separate forward-only observer**, not a modification to the four existing tmux services. It accepts ONLY new ENTRY decisions written by the existing Phase 4 virtual paper simulator **after observer initialization**, freezes their quote-side entries, then reads independently verified binary settlement outcomes **only after market close**.

PR #10's retrospective positive momentum hold ($0.563 on seven existing entries) is only a hypothesis. This module tests future, previously unseen markets. Do not infer an edge from the reviewed PR #10 cohort.

## Data flow

- Inputs (all SQLite `mode=ro` and `query_only=ON`): original public quote collector, existing virtual paper ledger, original predictor settled-outcomes DB.
- Output (separate new local SQLite file): `data/settlement_shadow/shadow.db`. No writes to any original DB, no network traffic, no credentials.
- Initial startup checkpoints `MAX(rowid)` of existing `paper_decisions`. Previously recorded paper trades and backfilled outcomes are **never enrolled**.
- Every later paper decision is handled once. Existing paper SKIP decisions are recorded as SOURCE_SKIP, not trades. Fresh ENTRY decisions must be observed within 25 seconds, have a matching existing `paper_trades` entry and exact original quote/source/timestamp/market-close provenance. Otherwise they are recorded as REJECTED. Crash/restart does not backfill any stale entry as a forward observation.
- Settlements are scored only after contract close, requiring matching explicit yes/no result and $1/$0 value, coherent close/pull times, and observation time not before the stored settlement pull time. Missing, inconsistent, and future-dated labels remain pending.
- Hypothetical held P&L is binary payout ($1 or $0) less the frozen ask, entry fee and entry slippage. No exit sale is implied. Quotes/fills and cost assumptions are unverified.
- Each original strategy remains separate; shared market events mean their P&Ls cannot be summed as a portfolio. Report coverage, fresh enrollment, rejected/stale decisions, pending labels, settlement wins/losses and after-assumed-cost virtual P&L by strategy.

## Validation only (current authorization)

Make a **new** isolated worktree for `feature/prospective-settlement-shadow`. Preserve existing worktrees and all four running processes. Run:

```bash
cd ~/Documents/kalshi-btc15m-settlement-shadow
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

Do NOT run the observer with production inputs yet. The first production invocation creates a separate ledger and establishes a prospective cursor, and repeated loop operation would introduce a fifth background process. Those actions require separate user approval after full test review. A test suite with temporary synthetic databases is sufficient for this validation request.

## Future launch (only with explicit approval)

Proposed command after separate authorization:

```bash
.venv/bin/python -m src.prospective_settlement_shadow --loop \
  --research-db /Users/tim/Documents/kalshi-btc-15m-research/data/research_stream.db \
  --paper-db /Users/tim/Documents/kalshi-btc15m-paper/data/paper_research/paper.db \
  --outcomes-db /Users/tim/Documents/kalshi-btc-15m-bot/data/kalshi_btc15m.db \
  --ledger-db data/settlement_shadow/shadow.db
```

Run separately (do not replace existing processes) and leave all actual trading flags disabled. Check status later with:

```bash
.venv/bin/python -m src.prospective_settlement_shadow --status \
  --ledger-db data/settlement_shadow/shadow.db
```

## Quantitative cautions

Positive forward-paper results remain virtual and do not prove executable fills. Check distinct settled markets, missing and rejected observations, market regimes, fee assumptions, calibration and uncertainty. Leave all six entry policies and their original exits unchanged; **no threshold tuning or model promotion** on this cohort. Any promotion or exchange trading requires separate evidence and explicit user approval.

**Hermes is validation/deployment only, never a source-code editor.** No edits, PR merge, restarts, DB modifications, new processes, automatic deployment or trading activation during validation.
