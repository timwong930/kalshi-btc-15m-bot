# Virtual-paper trade forensics (read-only)

This is an **offline diagnostic**, not a trading strategy, execution simulator, or exchange fill report. It does not change the live collector, simulator, model trainer, predictor, or their SQLite databases.

## Run from an isolated worktree

```bash
cd ~/Documents/kalshi-btc15m-forensics
.venv/bin/python -m src.paper_forensics \
  --paper-db /Users/tim/Documents/kalshi-btc15m-paper/data/paper_research/paper.db
```

No network connections or credentials are needed. SQLite opens the existing paper ledger with `mode=ro` and `PRAGMA query_only=ON`. The command returns JSON on stdout.

## Interpretation

- For each strategy, reconciles **observed quote-side gross price change** `exit_bid - entry_ask` with saved entry/exit fees, slippage, and recorded virtual P&L. If accounting does not reconcile within 1e-6 dollars, the command errors instead of reporting invalid profits.
- Reports trade counts and distinct closed markets; ENTRY/SKIP reasons; CLOSED/OPEN/CENSORED counts; exit reasons; aggregate gross quote movement and modeled costs. This can explain whether the modeled exit prices were already adverse before cost assumptions.
- Reprices the **same recorded closed entry/exit quote pairs** under illustrative per-side fees of $0.00, $0.01, $0.02, and $0.03 and slippage of $0.00, $0.01, and $0.02. Costs apply at **both** entry and exit.
- These scenarios are **conditional accounting**, not alternative prospective strategies or fee-accurate simulated replays. Changing costs would change stop/take-profit triggers and potentially skip/entry decisions. Holding exit quotes fixed cannot establish what trades would occur under changed costs.
- Open/censored positions have no reliable completed P&L and are excluded; selection bias remains possible. No recorded best bid/ask proves an actual exchange fill. The six strategies are separate one-contract counterfactuals, not a combined portfolio.
- Flat $0.02 fees and $0.01 slippage per side are Phase 4 research assumptions, not verified Kalshi BTC 15m transaction fees. Kalshi's published event-contract fee schedule is price-dependent: https://kalshi.com/docs/kalshi-fee-schedule.pdf . Verify the exact current contract fee treatment and applicable maker/taker conditions before claiming execution realism.

## Hermes acceptance steps

1. Fetch the new branch into an **additional** isolated worktree, never replacing the running paper simulator worktree.
2. Run the repository's **complete** `pytest -q` and `ruff check .`.
3. If checks pass, execute the above command; paste the full JSON output (no raw DB, secret or account data), or save sanitized JSON locally if too large.
4. Return concise observations and error traces for ChatGPT to address. **Do not edit code**, restart or replace tmux processes, merge PRs, alter databases, or enable trading.

The source branches and running processes remain unchanged when this diagnostic runs.
