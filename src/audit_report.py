"""Local, read-only audit of BTC research decisions. No trading API calls.

Settlement performance is COUNTERFACTUAL: it assumes one full contract
could have filled at the observed best ask, and uses the ASSUMED fee.
It is never realized P&L, an executed order, or proof of a trading edge.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import Counter
from pathlib import Path

from config.settings import load_settings

QUERY = """
SELECT a.*, s.result AS settled_result, s.settlement_value AS settlement_value
FROM edge_signal_audit AS a
LEFT JOIN settled_outcomes AS s ON s.ticker = a.ticker
"""


def open_audit_db(path: str) -> sqlite3.Connection:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Audit DB not found: {resolved}. Run --predict first.")
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='edge_signal_audit'"
    ).fetchone():
        conn.close()
        raise RuntimeError("No audit table yet; start the updated --predict bot to initialize it.")
    return conn


def counterfactual_pnl(row: sqlite3.Row) -> float | None:
    """Hypothetical one-contract buy/hold return, not an observed fill."""
    if row["action"] not in ("BUY UP", "BUY DOWN"):
        return None
    if row["settled_result"] not in ("yes", "no"):
        return None
    ask = row["candidate_ask"]
    if ask is None:
        return None
    won = (row["action"] == "BUY UP" and row["settled_result"] == "yes") or (
        row["action"] == "BUY DOWN" and row["settled_result"] == "no"
    )
    return (1.0 if won else 0.0) - ask - row["fee_assumed"]


def first_candidate_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Count at most one signal per 15-minute market to avoid correlated samples."""
    return conn.execute(
        QUERY + """
        WHERE a.id IN (
            SELECT MIN(id) FROM edge_signal_audit
            WHERE action IN ('BUY UP', 'BUY DOWN') GROUP BY ticker
        )
        ORDER BY a.id
        """
    ).fetchall()


def build_summary(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM edge_signal_audit").fetchone()[0]
    skips = conn.execute(
        "SELECT COUNT(*) FROM edge_signal_audit WHERE action='SKIP'"
    ).fetchone()[0]
    candidates = first_candidate_rows(conn)
    settled = [r for r in candidates if counterfactual_pnl(r) is not None]
    pnl = [counterfactual_pnl(r) for r in settled]
    wins = sum(
        (r["action"] == "BUY UP" and r["settled_result"] == "yes")
        or (r["action"] == "BUY DOWN" and r["settled_result"] == "no")
        for r in settled
    )
    return {
        "evaluations": total,
        "skips": skips,
        "candidate_evaluations": total - skips,
        "candidate_markets": len(candidates),
        "settled_candidate_markets": len(settled),
        "hypothetical_wins": wins,
        "hypothetical_win_rate": wins / len(settled) if settled else None,
        "hypothetical_pnl_per_one_contract_per_market": sum(pnl),
        "hypothetical_avg_pnl": sum(pnl) / len(pnl) if pnl else None,
    }


def export_csv(conn: sqlite3.Connection, output: str) -> int:
    """Full, append-only evaluation history plus outcomes; CSV contains no secrets."""
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cursor = conn.execute(QUERY + " ORDER BY a.id")
    columns = [d[0] for d in cursor.description]
    n = 0
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns + ["counterfactual_one_contract_pnl_if_filled"])
        for row in cursor:
            writer.writerow(list(row) + [counterfactual_pnl(row)])
            n += 1
    return n


def skip_reason_group(reason: str) -> str:
    """Normalize per-tick changing numeric details to auditable root causes."""
    if reason.startswith("best modeled edge "):
        return "modeled edge below configured threshold"
    if reason.startswith("BUY UP:") or reason.startswith("BUY DOWN:"):
        if "book ask " in reason:
            return "orderbook/snapshot price mismatch"
        if "best-price depth" in reason:
            return "insufficient top-of-book depth"
        if "no opposite-side resting bid" in reason:
            return "missing opposite-side resting bids"
    if reason.startswith("Kalshi quote is stale"):
        return "stale Kalshi quote"
    if reason.startswith("BTC spot is stale"):
        return "stale BTC spot"
    return reason


def print_report(conn: sqlite3.Connection, limit: int = 15) -> None:
    s = build_summary(conn)
    print("BTC 15m READ-ONLY SIGNAL AUDIT")
    print(f"Evaluations: {s['evaluations']}, SKIPs: {s['skips']}, candidate evaluations: {s['candidate_evaluations']}")
    print(
        f"Markets with candidate: {s['candidate_markets']}; "
        f"settled first-candidate markets: {s['settled_candidate_markets']}"
    )
    if s["settled_candidate_markets"]:
        print(
            f"COUNTERFACTUAL first-candidate win rate: {s['hypothetical_win_rate']:.1%}; "
            f"1-contract summed P&L (assumed fills/fees): "
            f"${s['hypothetical_pnl_per_one_contract_per_market']:+.4f}; "
            f"avg: ${s['hypothetical_avg_pnl']:+.4f}"
        )
    else:
        print("No settled candidate markets yet; no performance estimate.")
    print("NOT actual trades, observed fills, realized P&L, or independent observations per tick.")

    days = conn.execute(
        """
        SELECT substr(evaluated_at_utc, 1, 10) AS day, COUNT(*) AS evaluations,
               SUM(CASE WHEN action != 'SKIP' THEN 1 ELSE 0 END) AS candidates
        FROM edge_signal_audit GROUP BY day ORDER BY day DESC LIMIT 7
        """
    ).fetchall()
    print("\nLast 7 observed days:")
    for row in days:
        print(f"  {row['day']}: {row['evaluations']} evaluations, {row['candidates']} candidates")

    skipped = conn.execute(
        "SELECT reason FROM edge_signal_audit WHERE action='SKIP' ORDER BY id DESC LIMIT 500"
    ).fetchall()
    top_reasons = Counter(skip_reason_group(row["reason"]) for row in skipped).most_common(5)
    if top_reasons:
        print("\nTop recent SKIP reasons (up to 500 latest SKIPs):")
        for reason, n in top_reasons:
            print(f"  {n}x {reason}")

    rows = conn.execute(QUERY + " ORDER BY a.id DESC LIMIT ?", (limit,)).fetchall()
    print(f"\nLast {len(rows)} evaluations:")
    for row in rows:
        edge = row["estimated_net_edge"]
        edge_text = f"{edge:+.6f}" if edge is not None else "n/a"
        print(
            f"  {row['evaluated_at_utc']} {row['ticker']} {row['action']} "
            f"side={row['candidate_direction'] or '-'} edge={edge_text} "
            f"settled={row['settled_result'] or 'pending'} | {row['reason']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit recorded read-only BTC probability signals")
    parser.add_argument("--db", default=None, help="SQLite path (default DB_PATH)")
    parser.add_argument("--limit", type=int, default=15, help="Number of recent decisions to print")
    parser.add_argument("--csv", default=None, metavar="PATH", help="Export ALL evaluations and outcomes to CSV")
    args = parser.parse_args()
    path = args.db or load_settings().db_path
    conn = open_audit_db(path)
    try:
        print_report(conn, limit=max(0, args.limit))
        if args.csv:
            n = export_csv(conn, args.csv)
            print(f"\nExported {n} evaluations to {args.csv}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
