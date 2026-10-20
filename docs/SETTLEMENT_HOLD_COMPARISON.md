# Phase 5f — Hold to verified settlement (research only)

## Research question

PR #9's 60-second hold reduced losses for some original entries but remained net negative. All 18 near-expiry quote-exit candidates were censored by invalid books. Selling near close and **holding the binary contract to verified settlement** are different actions. Kalshi's own documentation describes $1 if a simple binary contract wins and $0 otherwise: https://help.kalshi.com/en/articles/13823844-portfolio-balances-and-positions.

Use the same existing prospective virtual entry decisions, without touching the running simulator, to perform a strictly retrospective, post-settlement comparison. This does NOT add a new live or paper strategy, does NOT use outcomes for entry decisions, and does NOT represent real exchange fills.

## Validation rules

- Verify the existing virtual entry against original quote ID, ticker, source, receive timestamp, side, ask and saved cost. Reject missing or inconsistent entries.
- Match the ticker with the research market's original scheduled close time.
- Read the original predictor's `settled_outcomes` database in SQLite **read-only** mode. Accept only an explicit `result=yes/no` paired with a non-NULL `settlement_value` of exactly 1.0/0.0 that agrees with result.
- Reject conflicting binary values, nonbinary outcomes, absent settlement metadata, suspicious market close-time discrepancies, and settlement data recorded before contract close. Never infer an unobserved result.
- Hypothetical held-to-settlement P&L for an **assumed one-contract entry** = $1 if that side wins, else $0, minus the recorded entry ask, entry fee and entry slippage. There is no quote-side exit in this scenario and no assumed exit trading fee. This is conditional accounting, **not** a verified fee quote or actual settlement transfer.
- OPEN or CENSORED original quote-exit positions can still have a separate hypothetical hold-to-settlement result if settlement has been independently recorded; count these distinctly. No missing settlement means a fabricated $0 or a loss.
- Show all entries, excluded reasons, distinct markets, wins/losses, summed conditional P&L and a paired comparison against the original policy only when that original trade was CLOSED. Never sum overlapping strategy results into a portfolio.

## Mac Studio read-only verification

Create a new isolated worktree for `feature/settlement-hold-research`, leave all running tmux windows/worktrees untouched. Run:

```bash
cd ~/Documents/kalshi-btc15m-settlement-hold
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m src.settlement_hold_comparison \
  --research-db /Users/tim/Documents/kalshi-btc-15m-research/data/research_stream.db \
  --paper-db /Users/tim/Documents/kalshi-btc15m-paper/data/paper_research/paper.db \
  --outcomes-db /Users/tim/Documents/kalshi-btc-15m-bot/data/kalshi_btc15m.db
```

Return the complete JSON including the excluded-reason counts. If the outcomes DB has many missing `settlement_value` fields, return that observation rather than changing the validation rules or manufacturing payouts.

## Safety / interpretation

**Hermes validates, not edits.** Do not modify code, DBs, models, branches, PRs, jobs, or deployed services; do not restart processes, deploy new jobs, promote models, merge, or enable any trading. ChatGPT handles fixes in draft PRs; user approves merges and future execution.

Results are retrospective and evaluated on markets that have already been reviewed in earlier research. **Even positive hypothetical settlement P&L would not prove a repeatable edge.** Entry ask fills are unconfirmed; fees are research assumptions; more independent untouched future markets would be needed. An end-of-market hold also changes downside exposure versus early stop strategies.
