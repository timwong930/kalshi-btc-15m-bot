"""Read-only, retrospective *fixed-exit* counterfactuals for prospective paper entries.

Uses the exact recorded historical virtual entries. Changes only when a
hypothetical exit is requested; no new signals, exchange fills, or orders.
Results are not prospective validation of these alternative exits.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

# Frozen, explicitly predeclared policies; NOT thresholds fitted to PR #8.
POLICIES = {"hold_60_seconds": "60s", "hold_to_preclose": "preclose"}
PRE_CLOSE_MS = 45_000
HOLD_60_MS = 60_000
MIN_HOLD_MS = 1_000
MAX_GAP_MS = 30_000
MAX_SPREAD = 0.08
MIN_DEPTH = 5
EXIT_FEE = 0.02
EXIT_SLIPPAGE = 0.01


def _readonly(name: str) -> sqlite3.Connection:
    path = Path(name).expanduser().resolve(strict=True)
    db = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def _float(value) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _valid_book(row: sqlite3.Row) -> bool:
    values = [_float(row[field]) for field in (
        "yes_bid", "yes_ask", "no_bid", "no_ask", "yes_depth", "no_depth"
    )]
    if not row["valid"] or any(v is None for v in values):
        return False
    yb, ya, nb, na, yd, nd = values
    return (0 < yb <= ya < 1 and 0 < nb <= na < 1
            and ya - yb <= MAX_SPREAD and na - nb <= MAX_SPREAD
            and min(yd, nd) >= MIN_DEPTH)


def _status(status: str, reason: str, **fields) -> dict:
    return {"status": status, "reason": reason, **fields}


def _original_entry(research: sqlite3.Connection, trade: sqlite3.Row):
    row = research.execute(
        "SELECT * FROM research_quotes WHERE id=?", (trade["opened_quote_id"],)
    ).fetchone()
    if row is None:
        return None
    if (row["ticker"] != trade["ticker"] or row["source"] != trade["quote_source"]
            or row["received_ms"] != trade["opened_ms"] or trade["side"] not in ("up", "down")):
        return None
    key = "yes_ask" if trade["side"] == "up" else "no_ask"
    ask = _float(row[key])
    entry = _float(trade["entry_ask"])
    fee = _float(trade["entry_fee"])
    slip = _float(trade["entry_slippage"])
    if (not _valid_book(row) or any(v is None for v in (ask, entry, fee, slip))
            or fee < 0 or slip < 0 or abs(ask - entry) > 1e-6):
        return None
    return row


def _evaluate(research: sqlite3.Connection, trade: sqlite3.Row, policy: str, now_ms: int) -> dict:
    """Conservative quote-only replay with original outage/censor boundaries."""
    entry = _original_entry(research, trade)
    if entry is None:
        return _status("UNVERIFIABLE", "original_entry_quote_not_verifiable")
    cutoff = (trade["opened_ms"] + HOLD_60_MS if policy == "hold_60_seconds"
              else trade["close_ms"] - PRE_CLOSE_MS)
    cutoff = max(cutoff, trade["opened_ms"] + MIN_HOLD_MS)
    if cutoff > trade["close_ms"]:
        return _status("UNVERIFIABLE", "exit_target_after_market_close")

    # Do not cross an original simulator outage/censoring boundary. For OPEN
    # trades also respect the original simulator's last actually observed quote.
    horizon = min(now_ms, trade["close_ms"])
    if trade["status"] in ("CENSORED", "OPEN"):
        last_observed = trade["last_quote_ms"]
        if last_observed is None:
            return _status("UNVERIFIABLE", "missing_original_observation_boundary")
        horizon = min(horizon, int(last_observed))

    previous_ms = trade["opened_ms"]
    bid_key = "yes_bid" if trade["side"] == "up" else "no_bid"
    rows = research.execute(
        "SELECT * FROM research_quotes WHERE ticker=? AND source=? AND id>? "
        "AND received_ms<=? ORDER BY id",
        (trade["ticker"], trade["quote_source"], trade["opened_quote_id"], horizon),
    )
    for row in rows:
        if row["received_ms"] < previous_ms:
            return _status("CENSORED", "non_monotone_source_quote_time")
        if row["received_ms"] - previous_ms > MAX_GAP_MS:
            return _status("CENSORED", "gap_exceeds_30_seconds")
        previous_ms = row["received_ms"]
        if not _valid_book(row):
            return _status("CENSORED", "invalid_book_before_target_exit")
        if row["received_ms"] < cutoff:
            continue
        bid = _float(row[bid_key])
        pnl = (bid - EXIT_FEE - EXIT_SLIPPAGE
               - trade["entry_ask"] - trade["entry_fee"] - trade["entry_slippage"])
        return _status(
            "CLOSED", "fixed_time_exit", virtual_pnl=round(pnl, 6),
            exit_quote_id=row["id"], exit_ms=row["received_ms"],
            elapsed_seconds=round((row["received_ms"] - trade["opened_ms"]) / 1000, 3),
        )

    if trade["status"] == "CENSORED":
        return _status("CENSORED", "original_censor_boundary_no_executable_exit")
    if trade["status"] == "OPEN":
        return _status("PENDING", "original_position_still_open_or_observation_limited")
    if now_ms < trade["close_ms"]:
        return _status("PENDING", "target_exit_quote_not_yet_observed")
    return _status("CENSORED", "no_valid_target_exit_before_close")


def analyze(research: sqlite3.Connection, paper: sqlite3.Connection, *, now_ms: int) -> dict:
    results = defaultdict(lambda: defaultdict(list))
    for trade in paper.execute("SELECT * FROM paper_trades ORDER BY opened_ms, id"):
        for policy in POLICIES:
            results[trade["strategy"]][policy].append((trade, _evaluate(research, trade, policy, now_ms)))

    summary = {}
    for name, policy_results in sorted(results.items()):
        summary[name] = {}
        for policy, rows in sorted(policy_results.items()):
            statuses = Counter(candidate["status"] for _, candidate in rows)
            reasons = Counter(
                candidate["reason"] for _, candidate in rows if candidate["status"] != "CLOSED"
            )
            closed = [(original, candidate) for original, candidate in rows
                      if candidate["status"] == "CLOSED"]
            paired = [
                (original, candidate) for original, candidate in closed
                if original["status"] == "CLOSED" and _float(original["pnl_per_contract"]) is not None
            ]
            paired_markets = {original["ticker"] for original, _ in paired}
            alt_pnl = sum(candidate["virtual_pnl"] for _, candidate in closed)
            alt_pair = sum(candidate["virtual_pnl"] for _, candidate in paired)
            original_pair = sum(float(original["pnl_per_contract"]) for original, _ in paired)
            summary[name][policy] = {
                "original_virtual_entries": len(rows),
                "candidate_status_counts": dict(sorted(statuses.items())),
                "candidate_censor_or_pending_reasons": dict(sorted(reasons.items())),
                "candidate_completed_market_count": len({t["ticker"] for t, _ in closed}),
                "candidate_completed_virtual_pnl": round(alt_pnl, 6),
                "original_closed_candidate_closed_pairs": len(paired),
                "distinct_paired_markets": len(paired_markets),
                "paired_original_virtual_pnl": round(original_pair, 6),
                "paired_candidate_virtual_pnl": round(alt_pair, 6),
                "paired_candidate_minus_original": round(alt_pair - original_pair, 6),
                "warning": (
                    "Closed-only comparisons exclude censored alternatives. "
                    "This is retrospective hypothetical quote replay, not live or prospective fills."
                ),
            }
    return {
        "mode": "READ_ONLY_RETROSPECTIVE_FIXED_EXIT_COMPARISON",
        "no_exchange_orders": True,
        "analysis_as_of_ms": now_ms,
        "candidate_policies": {
            "hold_60_seconds": "First eligible observed bid at least 60 seconds after original entry; no TP/SL",
            "hold_to_preclose": "First eligible observed bid within 45 seconds of market close; no TP/SL",
        },
        "strategies": summary,
        "limitations": [
            "These exit policies were NOT run prospectively, and results do not validate a strategy edge.",
            "Original entries are reused; candidate P&L is hypothetical bid execution, not exchange fills.",
            "Censored and pending candidates have no P&L; closed-only comparison can be biased.",
            "A prior original simulator CENSOR or OPEN state limits usable subsequent observations.",
            "Quotes lack processing-time evidence for alternative policies; absence of leakage is not proof of fills.",
            "The $0.02 fee and $0.01 slippage per exit are unverified research assumptions.",
            "Shared markets across strategies are NOT independent and aggregated P&L is not a portfolio.",
            "Do not choose a winning policy based on this small already-inspected sample.",
        ],
    }


def diagnose(research_path: str, paper_path: str, *, now_ms: int | None = None) -> dict:
    as_of = int(time.time() * 1000) if now_ms is None else now_ms
    with closing(_readonly(research_path)) as research:
        with closing(_readonly(paper_path)) as paper:
            return analyze(research, paper, now_ms=as_of)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrospective quote-only exit-policy sensitivity")
    parser.add_argument("--research-db", required=True)
    parser.add_argument("--paper-db", required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.research_db, args.paper_db), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
