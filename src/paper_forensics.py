"""Read-only cost attribution for already-closed *virtual* quote-based trades.

Reprices observed entries/exits without replaying decisions or implying fills.
Never writes to any database and never interacts with the exchange.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

# Each scenario changes only the accounting of completed historical virtual
# entry/exit pairs. It does not re-evaluate TP/SL triggers or unfilled orders.
FEE_SCENARIOS = (0.00, 0.01, 0.02, 0.03)
SLIPPAGE_SCENARIOS = (0.00, 0.01, 0.02)
COST_COLUMNS = ("entry_fee", "exit_fee", "entry_slippage", "exit_slippage")


def readonly(path: str) -> sqlite3.Connection:
    source = Path(path).expanduser().resolve(strict=True)
    connection = sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _amount(row: sqlite3.Row, name: str) -> float:
    value = row[name]
    if value is None:
        raise ValueError(f"Closed virtual trade {row['id']} has null {name}")
    amount = float(value)
    if not math.isfinite(amount):
        raise ValueError(f"Closed virtual trade {row['id']} has nonfinite {name}")
    return amount


def _summarize_closed(rows: list[sqlite3.Row]) -> dict:
    """Raises if any persisted virtual trade's P&L fails reconciliation."""
    contributions = []
    for trade in rows:
        entry = _amount(trade, "entry_ask")
        exit_bid = _amount(trade, "exit_bid")
        recorded = _amount(trade, "pnl_per_contract")
        costs = tuple(_amount(trade, field) for field in COST_COLUMNS)
        if any(value < 0 for value in costs):
            raise ValueError(f"Closed virtual trade {trade['id']} has negative cost")
        gross = exit_bid - entry
        actual_cost = sum(costs)
        if abs(gross - actual_cost - recorded) > 1e-6:
            raise ValueError(f"Closed virtual trade {trade['id']} does not reconcile")
        contributions.append((gross, recorded, actual_cost))

    gross_total = sum(row[0] for row in contributions)
    net_total = sum(row[1] for row in contributions)
    cost_total = sum(row[2] for row in contributions)
    scenarios = []
    for fee in FEE_SCENARIOS:
        for slip in SLIPPAGE_SCENARIOS:
            # Two sides; this is *conditional accounting*, not a fresh
            # execution backtest with re-triggered exits.
            hypothetical = [gross - 2 * (fee + slip) for gross, _, _ in contributions]
            scenarios.append({
                "fee_per_side": fee,
                "slippage_per_side": slip,
                "closed_trade_count": len(rows),
                "conditional_pnl_same_observed_exits": round(sum(hypothetical), 6),
                "conditional_profitable_exits": sum(pnl > 0 for pnl in hypothetical),
            })
    return {
        "closed_trades": len(rows),
        "unique_closed_market_tickers": len({row["ticker"] for row in rows}),
        "gross_quote_movement_dollars": round(gross_total, 6),
        "assumed_recorded_costs_dollars": round(cost_total, 6),
        "recorded_net_simulated_pnl_dollars": round(net_total, 6),
        "profitable_gross_quote_movements": sum(gross > 0 for gross, _, _ in contributions),
        "profitable_net_simulated_exits": sum(net > 0 for _, net, _ in contributions),
        "mean_round_trip_recorded_cost": round(cost_total / len(rows), 6) if rows else None,
        "mean_gross_quote_movement": round(gross_total / len(rows), 6) if rows else None,
        "exit_reasons": dict(sorted(Counter(row["reason"] or "unknown" for row in rows).items())),
        "cost_scenarios": scenarios,
    }


def analyze(db: sqlite3.Connection) -> dict:
    by_strategy = defaultdict(list)
    status_counts = defaultdict(Counter)
    decision_counts = defaultdict(Counter)
    for row in db.execute("SELECT * FROM paper_trades ORDER BY id"):
        status_counts[row["strategy"]][row["status"]] += 1
        if row["status"] == "CLOSED":
            by_strategy[row["strategy"]].append(row)
    for row in db.execute("SELECT strategy, decision, reason FROM paper_decisions"):
        decision_counts[row["strategy"]][f"{row['decision']}:{row['reason']}"] += 1
    result = {}
    for name in sorted(set(status_counts) | set(decision_counts)):
        result[name] = {
            **_summarize_closed(by_strategy[name]),
            "trade_status_counts": dict(sorted(status_counts[name].items())),
            "decision_reason_counts": dict(sorted(decision_counts[name].items())),
        }
    return {
        "mode": "READ_ONLY_VIRTUAL_TRADE_ATTRIBUTION",
        "no_exchange_orders": True,
        "strategies": result,
        "limitations": [
            "Quoted bid/ask prices and available book depth do not establish executable exchange fills.",
            "Each strategy is a separate one-contract counterfactual; do not sum strategy P&Ls as a portfolio.",
            "Censored/open trades are excluded from P&L, creating possible selection bias.",
            "Cost sensitivities hold observed exits fixed; different costs would change TP/SL triggers and decisions.",
            "Scenarios are illustrative, not a verified fee schedule for these particular contracts.",
        ],
    }


def diagnose(path: str) -> dict:
    with closing(readonly(path)) as db:
        return analyze(db)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only virtual trade P&L and cost attribution")
    parser.add_argument("--paper-db", required=True, help="Existing virtual paper SQLite ledger")
    args = parser.parse_args()
    print(json.dumps(diagnose(args.paper_db), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
