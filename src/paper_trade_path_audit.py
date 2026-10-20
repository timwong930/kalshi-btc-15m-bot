"""Read-only quote-path diagnostics for existing prospective virtual paper trades.

Reconstructs only observed collector quote paths between persisted entry/exit
quote IDs. It does not replay trading rules, invent fills, or modify ledgers.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from statistics import mean, median

MAX_BTC_AGE_MS = 10_000
MIN_HOLD_MS = 1_000
MAX_BOOK_SPREAD = 0.08
MIN_BOOK_DEPTH = 5.0
NET_STOP = 0.10


def _read_only(filename: str) -> sqlite3.Connection:
    path = Path(filename).expanduser().resolve(strict=True)
    db = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def _number(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _summary(values: list[float]) -> dict:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": round(min(values), 6),
        "median": round(median(values), 6),
        "mean": round(mean(values), 6),
        "max": round(max(values), 6),
    }


def _quote_ok(row: sqlite3.Row) -> bool:
    if not row["valid"]:
        return False
    vals = [_number(row[key]) for key in (
        "yes_bid", "yes_ask", "no_bid", "no_ask", "yes_depth", "no_depth"
    )]
    if any(v is None for v in vals):
        return False
    yb, ya, nb, na, yd, nd = vals
    return (0 < yb <= ya < 1 and 0 < nb <= na < 1
            and ya - yb <= MAX_BOOK_SPREAD and na - nb <= MAX_BOOK_SPREAD
            and min(yd, nd) >= MIN_BOOK_DEPTH)


def _btc_asof(db: sqlite3.Connection, timestamp: int) -> float | None:
    row = db.execute(
        "SELECT received_ms, price FROM research_prices "
        "WHERE source='coinbase' AND received_ms <= ? "
        "ORDER BY received_ms DESC, id DESC LIMIT 1",
        (timestamp,),
    ).fetchone()
    if row is None or timestamp - row["received_ms"] > MAX_BTC_AGE_MS:
        return None
    price = _number(row["price"])
    return price if price is not None and price > 0 else None


def _trade_path(
    db: sqlite3.Connection, trade: sqlite3.Row,
) -> tuple[dict | None, str | None]:
    if trade["side"] not in ("up", "down"):
        return None, "unknown_side"
    side = "yes" if trade["side"] == "up" else "no"
    entry = db.execute("SELECT * FROM research_quotes WHERE id=?", (trade["opened_quote_id"],)).fetchone()
    exit_quote = db.execute("SELECT * FROM research_quotes WHERE id=?", (trade["exit_quote_id"],)).fetchone()
    if entry is None or exit_quote is None:
        return None, "missing_entry_or_exit_quote"
    if (entry["ticker"] != trade["ticker"] or exit_quote["ticker"] != trade["ticker"]
            or entry["source"] != trade["quote_source"]
            or exit_quote["source"] != trade["quote_source"]):
        return None, "ticker_or_source_mismatch"
    if (entry["received_ms"] != trade["opened_ms"] or exit_quote["received_ms"] != trade["exit_ms"]
            or entry["id"] >= exit_quote["id"] or entry["received_ms"] >= exit_quote["received_ms"]):
        return None, "time_or_quote_order_mismatch"

    bid0 = _number(entry[f"{side}_bid"])
    ask0 = _number(entry[f"{side}_ask"])
    exit_bid = _number(exit_quote[f"{side}_bid"])
    stored_entry_ask = _number(trade["entry_ask"])
    stored_exit_bid = _number(trade["exit_bid"])
    costs = [_number(trade[key]) for key in (
        "entry_fee", "entry_slippage", "exit_fee", "exit_slippage"
    )]
    pnl = _number(trade["pnl_per_contract"])
    if (not _quote_ok(entry) or not _quote_ok(exit_quote)
            or any(val is None or val < 0 for val in costs)
            or any(val is None for val in (bid0, ask0, exit_bid, stored_entry_ask, stored_exit_bid, pnl))):
        return None, "invalid_entry_exit_book_or_ledger"
    if (abs(ask0 - stored_entry_ask) > 1e-6
            or abs(exit_bid - stored_exit_bid) > 1e-6
            or abs((exit_bid - ask0 - sum(costs)) - pnl) > 1e-6):
        return None, "entry_exit_price_or_pnl_mismatch"

    # Each intermediate observation must come from this ticker/source and be
    # strictly later by quote ID. Observed rows are NOT hypothetical fills.
    between = db.execute(
        "SELECT * FROM research_quotes WHERE ticker=? AND source=? "
        "AND id>? AND id<=? ORDER BY id",
        (trade["ticker"], trade["quote_source"], entry["id"], exit_quote["id"]),
    ).fetchall()
    if not between or between[-1]["id"] != exit_quote["id"]:
        return None, "incomplete_quote_path"
    previous_ms = entry["received_ms"]
    first_later_valid = None
    for observation in between:
        if observation["received_ms"] < previous_ms:
            return None, "non_monotonic_quote_times"
        previous_ms = observation["received_ms"]
        if (first_later_valid is None
                and observation["received_ms"] >= trade["opened_ms"] + MIN_HOLD_MS
                and _quote_ok(observation)):
            first_later_valid = observation

    btc0 = _btc_asof(db, trade["opened_ms"])
    btc1 = _btc_asof(db, trade["exit_ms"])
    btc_change_bps = 10000 * (btc1 / btc0 - 1) if btc0 and btc1 else None

    # Identity: observed gross quote P&L = -entry spread + exit-bid drift.
    # These are NOT exchange fills, only quotes.
    spread = ask0 - bid0
    bid_drift = exit_bid - bid0
    gross = exit_bid - ask0
    if abs(gross - (bid_drift - spread)) > 1e-9:
        return None, "gross_quote_move_does_not_reconcile"
    modeled_total_cost = sum(costs)
    first_later_stop = None
    first_later_ms = None
    if first_later_valid is not None:
        first_later_bid = _number(first_later_valid[f"{side}_bid"])
        if first_later_bid is not None:
            first_later_stop = first_later_bid - ask0 - modeled_total_cost <= -NET_STOP + 1e-9
            first_later_ms = first_later_valid["received_ms"] - trade["opened_ms"]

    return {
        "ticker": trade["ticker"],
        "side": trade["side"],
        "exit_reason": trade["reason"],
        "opened_quote_id": entry["id"],
        "exit_quote_id": exit_quote["id"],
        "time_to_exit_seconds": round((trade["exit_ms"] - trade["opened_ms"]) / 1000, 3),
        "observed_later_quote_count": len(between),
        "entry_spread": round(spread, 6),
        "exit_bid_change_from_entry_bid": round(bid_drift, 6),
        "gross_quote_movement": round(gross, 6),
        "recorded_round_trip_cost": round(modeled_total_cost, 6),
        "recorded_virtual_net_pnl": round(pnl, 6),
        "bid_drift_to_reach_net_stop_from_entry_bid": round(
            ask0 + modeled_total_cost - NET_STOP - bid0, 6
        ),
        "first_later_structurally_valid_quote_elapsed_ms": first_later_ms,
        "first_later_structurally_valid_quote_at_net_stop": first_later_stop,
        "coinbase_move_bps": round(btc_change_bps, 4) if btc_change_bps is not None else None,
        "coinbase_side_aligned_move_bps": (
            round(btc_change_bps * (1 if trade["side"] == "up" else -1), 4)
            if btc_change_bps is not None else None
        ),
    }, None


def analyze(research: sqlite3.Connection, paper: sqlite3.Connection, *,
            include_trades: bool = False) -> dict:
    by_strategy = defaultdict(lambda: {
        "status_counts": Counter(),
        "closed_reasons": Counter(),
        "unmatched_reasons": Counter(),
        "closed": [],
    })
    for trade in paper.execute("SELECT * FROM paper_trades ORDER BY id"):
        g = by_strategy[trade["strategy"]]
        g["status_counts"][trade["status"]] += 1
        if trade["status"] != "CLOSED":
            continue
        g["closed_reasons"][trade["reason"] or "unknown"] += 1
        path, reason = _trade_path(research, trade)
        if reason is not None:
            g["unmatched_reasons"][reason] += 1
        else:
            g["closed"].append(path)

    strategies = {}
    for name, g in sorted(by_strategy.items()):
        rows = g["closed"]
        stop_rows = [r for r in rows if r["exit_reason"] == "stop_loss"]
        early_stops = [
            r for r in stop_rows
            if r["first_later_structurally_valid_quote_at_net_stop"] is True
        ]
        coinbase_rows = [r for r in rows if r["coinbase_side_aligned_move_bps"] is not None]
        result = {
            "trade_status_counts": dict(sorted(g["status_counts"].items())),
            "closed_exit_reasons": dict(sorted(g["closed_reasons"].items())),
            "analyzed_closed_trades": len(rows),
            "unmatched_closed_trades": sum(g["unmatched_reasons"].values()),
            "unmatched_reasons": dict(sorted(g["unmatched_reasons"].items())),
            "distinct_analyzed_markets": len({r["ticker"] for r in rows}),
            "time_to_exit_seconds": _summary([r["time_to_exit_seconds"] for r in rows]),
            "observed_later_quote_count": _summary([r["observed_later_quote_count"] for r in rows]),
            "entry_spread": _summary([r["entry_spread"] for r in rows]),
            "exit_bid_change_from_entry_bid": _summary([
                r["exit_bid_change_from_entry_bid"] for r in rows
            ]),
            "gross_quote_movement": _summary([r["gross_quote_movement"] for r in rows]),
            "stops_at_first_later_structurally_valid_quote": len(early_stops),
            "stops_with_first_later_structurally_valid_quote_observed": sum(
                r["first_later_structurally_valid_quote_at_net_stop"] is not None for r in stop_rows
            ),
            "stops_with_exit_in_15_seconds": sum(r["time_to_exit_seconds"] <= 15 for r in stop_rows),
            "stops_with_exit_in_30_seconds": sum(r["time_to_exit_seconds"] <= 30 for r in stop_rows),
            "stops_with_exit_in_60_seconds": sum(r["time_to_exit_seconds"] <= 60 for r in stop_rows),
            "coinbase_side_aligned_change_bps": _summary([
                r["coinbase_side_aligned_move_bps"] for r in coinbase_rows
            ]),
            "coinbase_valid_pairs": len(coinbase_rows),
            "coinbase_missing_or_stale_pairs": len(rows) - len(coinbase_rows),
        }
        if include_trades:
            result["closed_trade_paths"] = rows
        strategies[name] = result

    return {
        "mode": "READ_ONLY_VIRTUAL_TRADE_QUOTE_PATH_AUDIT",
        "never_actual_fills": True,
        "strategies": strategies,
        "limitations": [
            "Only observed quote rows between original entry and exit IDs are analyzed; no future data.",
            "Coinbase reference prices are NOT the BRTI settlement index; side-aligned changes are descriptive.",
            "A structurally valid quote does not prove simulator processing-time freshness or real execution.",
            "Stops at the first later structurally valid quote do not prove there was an exchange fill.",
            "Censored/open trades have no completed-trade trajectory or imputed P&L.",
            "Do not treat individual quote samples as independent markets or tune to this small sample.",
        ],
    }


def diagnose(research_path: str, paper_path: str, *, include_trades: bool = False) -> dict:
    with closing(_read_only(research_path)) as research:
        with closing(_read_only(paper_path)) as paper:
            return analyze(research, paper, include_trades=include_trades)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only virtual entry-to-exit quote-path audit")
    parser.add_argument("--research-db", required=True)
    parser.add_argument("--paper-db", required=True)
    parser.add_argument("--include-trades", action="store_true", help="Include per-trade quote-path summaries")
    args = parser.parse_args()
    print(json.dumps(diagnose(args.research_db, args.paper_db, include_trades=args.include_trades),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
