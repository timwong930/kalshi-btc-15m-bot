"""Read-only hold-to-settlement counterfactual for recorded virtual paper entries.

Requires independently recorded binary settlement values after market close.
No orders, virtual entry replay, outcome-based signal selection, or DB writes.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

MAX_CLOSE_DISAGREEMENT_MS = 120_000


def _readonly(path: str) -> sqlite3.Connection:
    source = Path(path).expanduser().resolve(strict=True)
    db = sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True, timeout=20)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _utc_ms(value: str) -> int | None:
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            return None
        return int(stamp.astimezone(timezone.utc).timestamp() * 1000)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _entry_ok(research: sqlite3.Connection, trade: sqlite3.Row) -> str | None:
    quote = research.execute(
        "SELECT ticker, source, received_ms, valid, yes_ask, no_ask "
        "FROM research_quotes WHERE id=?", (trade["opened_quote_id"],)
    ).fetchone()
    if quote is None:
        return "missing_original_entry_quote"
    if (quote["ticker"] != trade["ticker"]
            or quote["source"] != trade["quote_source"]
            or quote["received_ms"] != trade["opened_ms"]):
        return "entry_quote_provenance_mismatch"
    if trade["side"] not in ("up", "down"):
        return "invalid_side"
    raw_ask = quote["yes_ask"] if trade["side"] == "up" else quote["no_ask"]
    ask, entry = _number(raw_ask), _number(trade["entry_ask"])
    fee, slip = _number(trade["entry_fee"]), _number(trade["entry_slippage"])
    if (not quote["valid"] or any(x is None for x in (ask, entry, fee, slip))
            or not 0 < ask < 1 or abs(ask - entry) > 1e-6
            or fee < 0 or slip < 0):
        return "invalid_entry_quote_or_cost"
    if not trade["opened_ms"] < trade["close_ms"]:
        return "entry_after_scheduled_market_close"
    return None


def _label(outcomes: sqlite3.Connection, ticker: str, close_ms: int):
    label = outcomes.execute(
        "SELECT result, settlement_value, close_time_utc, pulled_at_utc "
        "FROM settled_outcomes WHERE ticker=?", (ticker,)
    ).fetchone()
    if label is None:
        return None, "no_stored_settlement"
    result = label["result"]
    payout_yes = _number(label["settlement_value"])
    if result not in ("yes", "no") or payout_yes is None:
        return None, "missing_binary_settlement_value"
    if payout_yes not in (0.0, 1.0):
        return None, "nonbinary_settlement_value"
    if (result == "yes") != (payout_yes == 1.0):
        return None, "settlement_result_value_conflict"
    market_close = _utc_ms(label["close_time_utc"])
    fetched = _utc_ms(label["pulled_at_utc"])
    if market_close is None or fetched is None:
        return None, "invalid_settlement_timestamps"
    if abs(market_close - close_ms) > MAX_CLOSE_DISAGREEMENT_MS:
        return None, "settlement_close_time_disagrees"
    if fetched < max(close_ms, market_close):
        return None, "settlement_reported_before_close"
    return {"yes_payout": payout_yes, "pulled_at_ms": fetched}, None


def analyze(
    research: sqlite3.Connection, paper: sqlite3.Connection, outcomes: sqlite3.Connection,
) -> dict:
    by_strategy = defaultdict(lambda: {
        "status_counts": Counter(),
        "excluded_reasons": Counter(),
        "settled": [],
        "paired": [],
    })
    # Every recorded virtual entry is considered; no outcome-dependent entry
    # selection or fabricated fill. Label use is exclusively after settlement.
    for trade in paper.execute("SELECT * FROM paper_trades ORDER BY opened_ms, id"):
        g = by_strategy[trade["strategy"]]
        g["status_counts"][trade["status"]] += 1
        problem = _entry_ok(research, trade)
        if problem:
            g["excluded_reasons"][problem] += 1
            continue
        market = research.execute(
            "SELECT close_ms FROM research_markets WHERE ticker=?", (trade["ticker"],)
        ).fetchone()
        if market is None or market["close_ms"] != trade["close_ms"]:
            g["excluded_reasons"]["missing_or_mismatched_market_close"] += 1
            continue
        settlement, problem = _label(outcomes, trade["ticker"], trade["close_ms"])
        if problem:
            g["excluded_reasons"][problem] += 1
            continue

        side_wins = ((trade["side"] == "up") == (settlement["yes_payout"] == 1.0))
        payout = 1.0 if side_wins else 0.0
        # A held binary contract has no bid-side exit transaction. The saved
        # entry fee/slippage are unverified research assumptions; no settlement
        # fee assumed, consistent with ordinary binary payout rules.
        pnl = payout - trade["entry_ask"] - trade["entry_fee"] - trade["entry_slippage"]
        item = {
            "ticker": trade["ticker"],
            "side": trade["side"],
            "original_virtual_status": trade["status"],
            "original_entry_quote_id": trade["opened_quote_id"],
            "settlement_pulled_at_ms": settlement["pulled_at_ms"],
            "hypothetical_payout": payout,
            "hypothetical_hold_net_pnl": round(pnl, 6),
            "original_virtual_net_pnl": (
                _number(trade["pnl_per_contract"]) if trade["status"] == "CLOSED" else None
            ),
        }
        g["settled"].append(item)
        if trade["status"] == "CLOSED":
            original = _number(trade["pnl_per_contract"])
            if original is not None:
                g["paired"].append((pnl, original, trade["ticker"]))

    results = {}
    for name, g in sorted(by_strategy.items()):
        settled, paired = g["settled"], g["paired"]
        settled_pnl = sum(row["hypothetical_hold_net_pnl"] for row in settled)
        candidate_pair = sum(candidate for candidate, _, _ in paired)
        baseline_pair = sum(original for _, original, _ in paired)
        results[name] = {
            "original_status_counts": dict(sorted(g["status_counts"].items())),
            "recorded_entries": sum(g["status_counts"].values()),
            "validated_settlement_entries": len(settled),
            "validated_distinct_markets": len({row["ticker"] for row in settled}),
            "excluded_entries": sum(g["excluded_reasons"].values()),
            "excluded_reasons": dict(sorted(g["excluded_reasons"].items())),
            "settlement_winning_entries": sum(row["hypothetical_payout"] == 1.0 for row in settled),
            "settlement_losing_entries": sum(row["hypothetical_payout"] == 0.0 for row in settled),
            "hypothetical_hold_net_total": round(settled_pnl, 6),
            "hypothetical_hold_net_per_settled_entry": (
                round(settled_pnl / len(settled), 6) if settled else None
            ),
            "paired_original_closed_count": len(paired),
            "paired_distinct_markets": len({ticker for _, _, ticker in paired}),
            "paired_original_closed_pnl": round(baseline_pair, 6),
            "paired_hold_settlement_pnl": round(candidate_pair, 6),
            "paired_hold_minus_original": round(candidate_pair - baseline_pair, 6),
            "warning": (
                "Retrospective matched outcomes, not prospective simulated fills; "
                "no risk conclusion from a small, previously inspected sample."
            ),
        }
    return {
        "mode": "READ_ONLY_HOLD_TO_VERIFIED_SETTLEMENT_COUNTERFACTUAL",
        "actual_orders": False,
        "does_not_use_outcome_to_select_entry": True,
        "strategies": results,
        "limitations": [
            "All virtual entries came from existing selection policies; no counterfactual new entry signal.",
            "Only verified binary result plus consistent $0/$1 settlement value are accepted.",
            "These settlement outcomes were learned AFTER market close, never used as entry-time features.",
            "No exit order is modeled; actual entry at recorded ask remains unconfirmed.",
            "Assumed entry fees/slippage are not the verified contract trading fee schedule.",
            "Open/censored original quote-based trades MAY be settled in this distinct hold-to-settlement scenario.",
            "Do not sum separate strategy totals as one portfolio; some share the same market.",
            "Historical outcome comparison is not proof of prospective positive expected returns.",
        ],
    }


def diagnose(research_db: str, paper_db: str, outcomes_db: str) -> dict:
    with closing(_readonly(research_db)) as research:
        with closing(_readonly(paper_db)) as paper:
            with closing(_readonly(outcomes_db)) as outcomes:
                return analyze(research, paper, outcomes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only virtual hold-to-settlement comparison")
    parser.add_argument("--research-db", required=True)
    parser.add_argument("--paper-db", required=True)
    parser.add_argument("--outcomes-db", required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.research_db, args.paper_db, args.outcomes_db),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
