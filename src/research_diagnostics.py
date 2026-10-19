"""Read-only diagnostics for Kalshi 15-minute research and virtual paper ledgers.

Does not modify either database or import any live trading/collector modules.
This script is a measurement tool, not a trading signal or profitability model.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

# Keep aligned with the frozen Phase 4 paper Policy; diagnostics report these
# cutoffs explicitly so the measured book-eligibility definition is auditable.
ENTRY_START_MS = 300_000
ENTRY_END_MS = 240_000
MAX_SPREAD = 0.08
MIN_DEPTH = 5.0
GAP_MS = 30_000


def _readonly(path: str) -> sqlite3.Connection:
    file_path = Path(path).expanduser().resolve(strict=True)
    db = sqlite3.connect(f"{file_path.as_uri()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def _book_eligible(row: sqlite3.Row) -> bool:
    """Static spread/depth filter only; NOT proof of a timely or filled order."""
    if not row["valid"]:
        return False
    try:
        yb, ya, nb, na, yd, nd = (
            float(row[field]) for field in (
                "yes_bid", "yes_ask", "no_bid", "no_ask", "yes_depth", "no_depth"
            )
        )
        return (all(math.isfinite(v) for v in (yb, ya, nb, na, yd, nd))
                and 0 < yb <= ya < 1 and 0 < nb <= na < 1
                and ya - yb <= MAX_SPREAD and na - nb <= MAX_SPREAD
                and min(yd, nd) >= MIN_DEPTH)
    except (TypeError, ValueError):
        return False


def inspect_research(db: sqlite3.Connection, *, now_ms: int) -> dict:
    markets = {
        row["ticker"]: dict(row)
        for row in db.execute(
            "SELECT ticker, close_ms, first_seen_ms FROM research_markets"
        )
    }
    # A completed observation window is not necessarily a complete market.
    windows = {
        ticker: {"seen": False, "eligible": False}
        for ticker, m in markets.items()
        if m["close_ms"] - ENTRY_END_MS <= now_ms
    }
    counts = Counter()
    global_gaps = Counter()
    per_market_gaps = Counter()
    last_global = None
    last_key = {}
    max_global_ms = 0
    max_per_market_ms = 0
    for row in db.execute(
        "SELECT id, ticker, source, received_ms, valid, yes_bid, yes_ask, "
        "no_bid, no_ask, yes_depth, no_depth "
        "FROM research_quotes ORDER BY received_ms, id"
    ):
        counts["rows"] += 1
        counts[f"source:{row['source']}"] += 1
        if row["valid"]:
            counts["valid_flag_true"] += 1
        eligible = _book_eligible(row)
        if eligible:
            counts["passes_static_book_filter"] += 1
        key = (row["ticker"], row["source"])
        stamp = row["received_ms"]
        if last_global is not None:
            delta = stamp - last_global[0]
            max_global_ms = max(max_global_ms, delta)
            if delta > GAP_MS:
                global_gaps["over_30s"] += 1
                if row["ticker"] != last_global[1]:
                    global_gaps["different_ticker"] += 1
                elif row["source"] != last_global[2]:
                    global_gaps["same_ticker_different_source"] += 1
                else:
                    global_gaps["same_ticker_same_source"] += 1
        last_global = (stamp, row["ticker"], row["source"])
        previous = last_key.get(key)
        if previous is not None:
            delta = stamp - previous
            max_per_market_ms = max(max_per_market_ms, delta)
            if delta > GAP_MS:
                per_market_gaps["over_30s"] += 1
                per_market_gaps[f"source:{row['source']}"] += 1
        last_key[key] = stamp
        window = windows.get(row["ticker"])
        market = markets.get(row["ticker"])
        if window is not None and market is not None:
            remaining = market["close_ms"] - stamp
            if ENTRY_END_MS <= remaining <= ENTRY_START_MS:
                window["seen"] = True
                window["eligible"] |= eligible

    window_counts = Counter()
    for ticker, state in windows.items():
        m = markets[ticker]
        window_counts["completed_windows"] += 1
        if m["first_seen_ms"] > m["close_ms"] - ENTRY_START_MS:
            window_counts["first_discovered_after_window_open"] += 1
        if state["eligible"]:
            window_counts["has_static_book_eligible_quote"] += 1
        elif state["seen"]:
            window_counts["quotes_present_none_book_eligible"] += 1
        else:
            window_counts["no_quote_in_entry_window"] += 1

    return {
        "research_markets": len(markets),
        "quotes": dict(sorted(counts.items())),
        "global_consecutive_gaps": {
            **dict(sorted(global_gaps.items())), "max_seconds": round(max_global_ms / 1000, 3),
        },
        "within_same_ticker_source_gaps": {
            **dict(sorted(per_market_gaps.items())),
            "max_seconds": round(max_per_market_ms / 1000, 3),
        },
        "entry_window_coverage": dict(sorted(window_counts.items())),
        "static_filter": {"min_depth": MIN_DEPTH, "max_spread": MAX_SPREAD,
                          "entry_seconds_before_close": [300, 240]},
        "limitations": [
            "Global quote gaps across tickers are not evidence of within-market outages.",
            "Within-market gaps may reflect inactive books or sparse updates, not necessarily collector failures.",
            ("Static book-eligible means only spread/depth/price checks passed; "
             "it does not confirm quote freshness, settlement eligibility, available fill, "
             "or a simulated trade."),
            "A market discovered after the entry window opened may have partial coverage.",
        ],
    }


def inspect_paper(db: sqlite3.Connection) -> dict:
    cursor_row = db.execute(
        "SELECT value FROM paper_meta WHERE key='last_quote_id'"
    ).fetchone()
    decisions = [
        {"strategy": r[0], "decision": r[1], "reason": r[2], "count": r[3]}
        for r in db.execute(
            "SELECT strategy, decision, reason, COUNT(*) FROM paper_decisions "
            "GROUP BY strategy, decision, reason ORDER BY strategy, decision, reason"
        )
    ]
    outcomes = [
        {"strategy": r[0], "status": r[1], "reason": r[2], "count": r[3]}
        for r in db.execute(
            "SELECT strategy, status, reason, COUNT(*) FROM paper_trades "
            "GROUP BY strategy, status, reason ORDER BY strategy, status, reason"
        )
    ]
    return {"last_quote_id": int(cursor_row[0]) if cursor_row else None,
            "decision_reasons": decisions, "trade_status_reasons": outcomes,
            "note": "Counts describe virtual decisions, not real orders or fills."}


def diagnose(research_path: str, paper_path: str | None = None,
             *, now_ms: int | None = None) -> dict:
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    with closing(_readonly(research_path)) as source:
        research = inspect_research(source, now_ms=now_ms)
    paper = None
    if paper_path is not None:
        with closing(_readonly(paper_path)) as ledger:
            paper = inspect_paper(ledger)
    return {"as_of_epoch_ms": now_ms,
            "research": research, "paper": paper,
            "read_only": True, "actual_orders": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only per-market quote and virtual-paper diagnostic")
    parser.add_argument("--research-db", required=True)
    parser.add_argument("--paper-db", help="Optional existing virtual paper ledger")
    args = parser.parse_args()
    print(json.dumps(diagnose(args.research_db, args.paper_db), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
