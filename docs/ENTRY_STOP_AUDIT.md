# Entry bid/ask spread versus simulated net stop — read-only audit

This diagnostic checks the specific concern raised by PR #6: with Phase 4's **$0.02 fee + $0.01 slippage on each side**, a **$0.10 net stop** can be crossed even if the contract's **bid is unchanged** after its virtual ask-side entry. It makes no trading decisions, does not revise stop/target rules and never places exchange orders.

## Why this matters

For a virtual buy at entry ask A, a hypothetical later virtual sale at bid B, and round-trip modeled costs C:

- Net P&L per one-contract counterfactual: **B - A - C**.
- Net take-profit at +$0.10 requires **B - A >= $0.16** for C=$0.06.
- Net stop-loss at -$0.10 requires **B - A <= -$0.04** for C=$0.06.
- At the *entry timestamp*, the market's same-side bid B0 may already be more than $0.04 below A. Then **B0 - A - C <= -$0.10** even with no subsequent change in bid.

For example, ask $0.30, bid $0.25, fees/slippage $0.06 in total: initial liquidation mark = **-$0.11**. To break even, a later bid must rise **$0.11** from the entry bid; to hit +$0.10 net TP, it must rise **$0.21**.

This is a static *entry quote* mark, **not an observed later exit**. Phase 4 requires a new valid quote after at least 1 second and censors delayed/unusable books. The diagnostic does not claim any real fill could occur.

## Run on the Mac Studio after tests pass

Create an additional isolated worktree for \`feature/entry-stop-feasibility-audit\`. Do not overwrite the existing worktrees or disturb any tmux windows. Then:

\`\`\`bash
cd ~/Documents/kalshi-btc15m-entry-audit
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m src.entry_stop_audit \
  --research-db /Users/tim/Documents/kalshi-btc-15m-research/data/research_stream.db \
  --paper-db /Users/tim/Documents/kalshi-btc15m-paper/data/paper_research/paper.db
\`\`\`

Both databases are opened with SQLite \`mode=ro\` and \`PRAGMA query_only=ON\`. No schema changes, raw data uploads, direct network requests, order paths, or automatic model promotion.

The JSON report groups by paper strategy and includes:

- Recorded entry count, strict original quote-ID/ticker/source matching, and unmatched reason counts.
- Mean/range of initial spread, immediate *theoretical* net mark at the entry bid, and implied rise in subsequent bid required to break even or reach take-profit.
- Count/rate of entries where **unchanged entry bid would already cross the net stop**, with an exit-reason cross-tab for already-closed observations.
- Counts for \`OPEN\`, \`CENSORED\`, and \`CLOSED\`; no invented values for censored trade P&L.
- Number of entries whose **stored** entry fee/slippage differ from current default Phase 4 \`Policy\`, in case the running simulator diverged.

Interpret as a possible explanation for mechanically frequent stops, **not** proof of lost actual money, profitable opportunity, or model failure. Costs are the research policy's assumptions, not confirmed Kalshi BTC-specific fees. Do not retune stop/TP values using these same 19 observed markets and call them validated.

## Hermes validation only

Hermes should fetch the draft branch, run full pytest/Ruff, execute the read-only CLI, and return complete JSON. Do **not** modify source, restart any collector/trainer/simulator/predictor process, touch databases, change flags, deploy new services, merge PRs or enable trading. All changes remain ChatGPT-owned.
