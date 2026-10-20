# Phase 5e — Fixed exit counterfactuals (retrospective only)

## Question

PR #8 found rapid observed bid-price declines after entry, while PR #7 ruled out the hypothesis that entry spread alone mechanically crossed the stop. Before changing a deployed trading policy, test whether **predeclared time exits** would have behaved differently on the *same* originally recorded virtual entries.

## Two fixed alternatives

- `hold_60_seconds`: no stop or take-profit; hypothetically exit at the first observed, structurally valid same-source bid at least 60 seconds after the original virtual entry.
- `hold_to_preclose`: no stop or take-profit; hypothetically exit at the first observed, structurally valid same-source bid within 45 seconds of contract close.

Both alternatives preserve the original entry quote and original entry costs. For their exits, both use Phase 4's unverified **$0.02 fee plus $0.01 slippage per side**. Quotes must be temporally ordered, have consecutive gaps of no more than 30 seconds, and satisfy Phase 4 book spread/depth constraints. Source problems censor the candidate. Original `CENSORED` and `OPEN` trade boundaries prevent replaying price movement the paper process did not observe.

## Important limitations

This is a **retrospective quote-path counterfactual**, not prospective shadow trading or an exchange paper order. Reusing entries and holding exit conditions fixed does not establish execution, fillability, or profitability. Original and alternative exits represent different risk exposures and a policy that might have changed entry decisions in practice.

Report `CLOSED`, `CENSORED`, `PENDING`, and `UNVERIFIABLE` candidates separately. Candidate P&L excludes anything not closed. Paired differences compare only entries closed under BOTH original and alternative policies; this is a selected subsample and can be biased. Some separate strategies share the same 15-minute market. **Do not sum strategy totals into portfolio P&L or select a winning exit policy from these already-inspected markets.** A future unseen prospective cohort is required.

## Hermes verification-only runbook

Use a NEW isolated worktree for `feature/fixed-exit-counterfactuals` based on PR #8; do not switch or modify any running worktree.

```bash
cd ~/Documents/kalshi-btc15m-fixed-exits
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m src.fixed_exit_comparison \
  --research-db /Users/tim/Documents/kalshi-btc-15m-research/data/research_stream.db \
  --paper-db /Users/tim/Documents/kalshi-btc15m-paper/data/paper_research/paper.db
```

Return complete JSON and all failing test output if any. **Hermes must not edit code, alter DBs, restart processes, merge PRs, deploy services, or enable actual trading.** ChatGPT authors and fixes code; user approves future deployment and merges.
