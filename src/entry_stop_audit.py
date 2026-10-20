"""Read-only audit: does entry spread plus modeled costs already breach the stop?

This is a static, entry-time mark-to-bid thought experiment only.
It does NOT replay exit ticks, infer exchange fills, or modify the simulator.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from statistics import mean

from src.paper_sim import Policy


def _readonly(path: str) -> sqlite3.Connection:
    source = Path(path).expanduser().resolve(strict=True)
    db = sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def _finite_nonnegative(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": round(mean(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def analyze(
    research: sqlite3.Connection, paper: sqlite3.Connection, policy: Policy | None = None,
) -> dict:
    """Join each virtual entry to its exact recorded quote ID; no lookahead."""
    policy = Policy() if policy is None else policy
    groups = defaultdict(lambda: {
        "status_counts": Counter(),
        "exit_reason_counts": Counter(),
        "unmatched_reasons": Counter(),
        "spreads": [],
        "initial_marks": [],
        "break_even_moves": [],
        "take_profit_moves": [],
        "stop_on_unchanged_bid": 0,
        "stop_on_unchanged_bid_by_exit_reason": Counter(),
        "recorded_cost_mismatch": 0,
        "matched": 0,
        "total": 0,
    })
    for trade in paper.execute(
        "SELECT id, ticker, strategy, quote_source, side, opened_quote_id, "
        "entry_ask, entry_fee, entry_slippage, status, reason FROM paper_trades ORDER BY id"
    ):
        group = groups[trade["strategy"]]
        group["total"] += 1
        group["status_counts"][trade["status"]] += 1
        if trade["status"] == "CLOSED":
            group["exit_reason_counts"][trade["reason"] or "unknown"] += 1

        quote = research.execute(
            "SELECT id, ticker, source, yes_bid, yes_ask, no_bid, no_ask "
            "FROM research_quotes WHERE id=?", (trade["opened_quote_id"],)
        ).fetchone()
        if quote is None:
            group["unmatched_reasons"]["missing_source_quote"] += 1
            continue
        if quote["ticker"] != trade["ticker"] or quote["source"] != trade["quote_source"]:
            group["unmatched_reasons"]["ticker_or_source_mismatch"] += 1
            continue
        if trade["side"] not in ("up", "down"):
            group["unmatched_reasons"]["invalid_trade_side"] += 1
            continue
        prefix = "yes" if trade["side"] == "up" else "no"
        bid = _finite_nonnegative(quote[f"{prefix}_bid"])
        ask = _finite_nonnegative(quote[f"{prefix}_ask"])
        entry_ask = _finite_nonnegative(trade["entry_ask"])
        fee = _finite_nonnegative(trade["entry_fee"])
        slippage = _finite_nonnegative(trade["entry_slippage"])
        if None in (bid, ask, entry_ask, fee, slippage):
            group["unmatched_reasons"]["invalid_or_missing_price_or_cost"] += 1
            continue
        if not (0 < bid <= ask < 1 and abs(ask - entry_ask) <= 1e-6):
            group["unmatched_reasons"]["invalid_or_mismatched_entry_quote"] += 1
            continue

        group["matched"] += 1
        if (abs(fee - policy.fee_per_side) > 1e-6
                or abs(slippage - policy.slippage_per_side) > 1e-6):
            group["recorded_cost_mismatch"] += 1

        # Source of exit-side fee/slippage is the default frozen Phase 4
        # Policy, not an exchange schedule or an observed exchange fill.
        roundtrip_cost = fee + slippage + policy.fee_per_side + policy.slippage_per_side
        spread = ask - bid
        initial_mark = bid - entry_ask - roundtrip_cost
        # How much must the observed bid rise from its *entry-time bid*
        # to break even or reach the simulator's net take-profit?
        break_even_move = entry_ask + roundtrip_cost - bid
        take_profit_move = break_even_move + policy.take_profit
        already_stopped = initial_mark <= -policy.stop_loss + 1e-9

        group["spreads"].append(spread)
        group["initial_marks"].append(initial_mark)
        group["break_even_moves"].append(break_even_move)
        group["take_profit_moves"].append(take_profit_move)
        if already_stopped:
            group["stop_on_unchanged_bid"] += 1
            if trade["status"] == "CLOSED":
                group["stop_on_unchanged_bid_by_exit_reason"][trade["reason"] or "unknown"] += 1

    results = {}
    for strategy, g in sorted(groups.items()):
        results[strategy] = {
            "recorded_entries": g["total"],
            "matched_entry_quotes": g["matched"],
            "unmatched_entry_quotes": g["total"] - g["matched"],
            "unmatched_reasons": dict(sorted(g["unmatched_reasons"].items())),
            "status_counts": dict(sorted(g["status_counts"].items())),
            "closed_exit_reasons": dict(sorted(g["exit_reason_counts"].items())),
            "entry_spread_dollars": _summarize(g["spreads"]),
            "initial_net_mark_at_unchanged_bid_dollars": _summarize(g["initial_marks"]),
            "bid_rise_needed_from_entry_bid_to_break_even": _summarize(g["break_even_moves"]),
            "bid_rise_needed_from_entry_bid_to_net_take_profit": _summarize(g["take_profit_moves"]),
            "already_meets_net_stop_if_entry_bid_unchanged": g["stop_on_unchanged_bid"],
            "already_meets_net_stop_rate": (
                round(g["stop_on_unchanged_bid"] / g["matched"], 6) if g["matched"] else None
            ),
            "already_meets_net_stop_closed_exit_reasons": dict(
                sorted(g["stop_on_unchanged_bid_by_exit_reason"].items())
            ),
            "entry_costs_differ_from_current_policy": g["recorded_cost_mismatch"],
        }
    return {
        "mode": "READ_ONLY_ENTRY_SPREAD_STOP_FEASIBILITY",
        "not_real_fills": True,
        "policy": {
            "stop_loss_net_dollars": policy.stop_loss,
            "take_profit_net_dollars": policy.take_profit,
            "assumed_exit_fee_per_contract": policy.fee_per_side,
            "assumed_exit_slippage_per_contract": policy.slippage_per_side,
            "min_hold_ms": policy.min_hold_ms,
        },
        "strategies": results,
        "limitations": [
            "Entry bid is observed at the SAME timestamp as entry ask; it is not an actual exit.",
            "A stop at unchanged entry bid means the first later eligible quote COULD stop with no bid rise; it does not prove the later bid was unchanged.",
            "The simulator requires a later quote at least min_hold_ms after entry and may censor invalid or delayed quotes.",
            "This uses the stored entry cost and current Phase 4 Policy exit-side assumptions, not verified Kalshi contract fees.",
            "Open and censored trades are included for entry-time spread analysis, but have no inferred exit P&L.",
            "This is descriptive on already-observed data. Do not optimize thresholds against this same sample and claim out-of-sample profitability.",
        ],
    }


def diagnose(research_path: str, paper_path: str) -> dict:
    with closing(_readonly(research_path)) as research:
        with closing(_readonly(paper_path)) as paper:
            return analyze(research, paper)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only paper-entry spread/stop feasibility audit; never places trades",
    )
    parser.add_argument("--research-db", required=True)
    parser.add_argument("--paper-db", required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.research_db, args.paper_db), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
