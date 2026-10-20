"""Forward-only, virtual settlement-shadow observer. NEVER submits exchange orders.

Observes only newly committed decisions from the existing virtual paper DB,
freezes their original entry quotes, then adjudicates AFTER verified settlement.
Existing predictor/collector/trainer/paper simulator are completely untouched.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import sqlite3
import time
from collections import Counter
from contextlib import closing, contextmanager
from pathlib import Path

from src.settlement_hold_comparison import _entry_ok, _label, _readonly

LOG = logging.getLogger("settlement_shadow")
MAX_CAPTURE_DELAY_MS = 25_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_meta(
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shadow_intake(
    decision_rowid INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    strategy TEXT NOT NULL,
    paper_decision TEXT NOT NULL,
    event_ms INTEGER NOT NULL,
    observed_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shadow_entries(
    decision_rowid INTEGER PRIMARY KEY REFERENCES shadow_intake(decision_rowid),
    ticker TEXT NOT NULL,
    strategy TEXT NOT NULL,
    quote_id INTEGER NOT NULL,
    entry_ms INTEGER NOT NULL,
    close_ms INTEGER NOT NULL,
    side TEXT NOT NULL,
    entry_ask REAL NOT NULL,
    entry_fee REAL NOT NULL,
    entry_slippage REAL NOT NULL,
    state TEXT NOT NULL DEFAULT 'WAITING_FOR_SETTLEMENT',
    settlement_pulled_at_ms INTEGER,
    settlement_observed_ms INTEGER,
    payout REAL,
    virtual_net_pnl REAL,
    last_label_problem TEXT,
    UNIQUE(ticker, strategy)
);
CREATE INDEX IF NOT EXISTS idx_shadow_unresolved ON shadow_entries(state, close_ms);
"""


def initialize_ledger(path: str, paper: sqlite3.Connection, *, now_ms: int) -> sqlite3.Connection:
    """First use checkpoints the latest paper decision: never retroactive entries."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(output), timeout=20)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    saved = db.execute(
        "SELECT value FROM shadow_meta WHERE key='last_decision_rowid'"
    ).fetchone()
    if saved is None:
        if db.execute("SELECT COUNT(*) FROM shadow_entries").fetchone()[0]:
            db.close()
            raise ValueError("Existing uncheckpointed shadow ledger: refusing to replay")
        cursor = paper.execute("SELECT COALESCE(MAX(rowid), 0) FROM paper_decisions").fetchone()[0]
        with db:
            db.executemany("INSERT INTO shadow_meta(key, value) VALUES (?, ?)", [
                ("last_decision_rowid", str(cursor)),
                ("starting_decision_rowid", str(cursor)),
                ("created_epoch_ms", str(now_ms)),
            ])
    return db


def _capture(
    research: sqlite3.Connection,
    paper: sqlite3.Connection,
    output: sqlite3.Connection,
    decision: sqlite3.Row,
    now_ms: int,
) -> None:
    rowid = decision["source_rowid"]
    decision_type = decision["decision"]
    if decision_type != "ENTRY":
        status, reason = "SOURCE_SKIP", decision["reason"]
    elif not 0 <= now_ms - decision["event_ms"] <= MAX_CAPTURE_DELAY_MS:
        status, reason = "REJECTED", "decision_not_observed_promptly"
    else:
        original = paper.execute(
            "SELECT * FROM paper_trades WHERE ticker=? AND strategy=?",
            (decision["ticker"], decision["strategy"]),
        ).fetchone()
        if original is None:
            status, reason = "REJECTED", "no_matching_original_paper_entry"
        elif (decision["quote_id"] != original["opened_quote_id"]
              or decision["event_ms"] != original["opened_ms"]):
            status, reason = "REJECTED", "decision_entry_provenance_mismatch"
        else:
            why = _entry_ok(research, original)
            market = research.execute(
                "SELECT close_ms FROM research_markets WHERE ticker=?",
                (original["ticker"],),
            ).fetchone()
            if why:
                status, reason = "REJECTED", why
            elif market is None or market["close_ms"] != original["close_ms"]:
                status, reason = "REJECTED", "missing_or_inconsistent_market_close"
            else:
                status, reason = "ENROLLED", "fresh_original_paper_entry"

    output.execute(
        "INSERT OR IGNORE INTO shadow_intake VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (rowid, decision["ticker"], decision["strategy"], decision_type,
         decision["event_ms"], now_ms, status, reason),
    )
    if status == "ENROLLED":
        output.execute(
            """INSERT OR IGNORE INTO shadow_entries(
                decision_rowid, ticker, strategy, quote_id, entry_ms, close_ms,
                side, entry_ask, entry_fee, entry_slippage, state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'WAITING_FOR_SETTLEMENT')""",
            (rowid, original["ticker"], original["strategy"], original["opened_quote_id"],
             original["opened_ms"], original["close_ms"], original["side"],
             original["entry_ask"], original["entry_fee"], original["entry_slippage"]),
        )


def _settle_pending(outcomes: sqlite3.Connection, output: sqlite3.Connection, now_ms: int) -> None:
    rows = output.execute(
        "SELECT * FROM shadow_entries WHERE state='WAITING_FOR_SETTLEMENT' AND close_ms<?",
        (now_ms,),
    ).fetchall()
    for entry in rows:
        settlement, reason = _label(outcomes, entry["ticker"], entry["close_ms"])
        if reason is not None:
            output.execute(
                "UPDATE shadow_entries SET last_label_problem=? WHERE decision_rowid=?",
                (reason, entry["decision_rowid"]),
            )
            continue
        # The label must be *known* by the observation timestamp. A wrongly
        # timestamped future label cannot be applied to this research cycle.
        if settlement["pulled_at_ms"] > now_ms:
            output.execute(
                "UPDATE shadow_entries SET last_label_problem=? WHERE decision_rowid=?",
                ("settlement_record_in_future", entry["decision_rowid"]),
            )
            continue
        payout = float(
            (entry["side"] == "up") == (settlement["yes_payout"] == 1.0)
        )
        net = (payout - entry["entry_ask"] - entry["entry_fee"]
               - entry["entry_slippage"])
        output.execute(
            """UPDATE shadow_entries
               SET state='SETTLED', settlement_pulled_at_ms=?,
                   settlement_observed_ms=?, payout=?, virtual_net_pnl=?,
                   last_label_problem=NULL
               WHERE decision_rowid=? AND state='WAITING_FOR_SETTLEMENT'""",
            (settlement["pulled_at_ms"], now_ms, payout, round(net, 6),
             entry["decision_rowid"]),
        )


def tick(
    research: sqlite3.Connection,
    paper: sqlite3.Connection,
    outcomes: sqlite3.Connection,
    output: sqlite3.Connection,
    *,
    now_ms: int | None = None,
) -> dict:
    """One idempotent forward cursor step, with settlement only after close."""
    observed = int(time.time() * 1000) if now_ms is None else now_ms
    cursor = int(output.execute(
        "SELECT value FROM shadow_meta WHERE key='last_decision_rowid'"
    ).fetchone()[0])
    decisions = paper.execute(
        "SELECT rowid AS source_rowid, ticker, strategy, event_ms, quote_id, "
        "decision, reason FROM paper_decisions WHERE rowid>? ORDER BY rowid LIMIT 1000",
        (cursor,),
    ).fetchall()
    for decision in decisions:
        with output:
            _capture(research, paper, output, decision, observed)
            output.execute(
                "UPDATE shadow_meta SET value=? WHERE key='last_decision_rowid'",
                (str(decision["source_rowid"]),),
            )
    with output:
        _settle_pending(outcomes, output, observed)
    return {"new_paper_decisions": len(decisions),
            "last_decision_rowid": decisions[-1]["source_rowid"] if decisions else cursor}


def report(output: sqlite3.Connection) -> dict:
    strategies = {}
    names = {
        row[0] for row in output.execute("SELECT DISTINCT strategy FROM shadow_intake")
    }
    for name in sorted(names):
        intakes = Counter(
            (row["status"], row["reason"])
            for row in output.execute("SELECT status, reason FROM shadow_intake WHERE strategy=?", (name,))
        )
        entries = output.execute(
            "SELECT ticker, state, payout, virtual_net_pnl, last_label_problem "
            "FROM shadow_entries WHERE strategy=? ORDER BY decision_rowid",
            (name,),
        ).fetchall()
        settled = [row for row in entries if row["state"] == "SETTLED"]
        total_net = sum(row["virtual_net_pnl"] for row in settled)
        strategies[name] = {
            "paper_intake_reasons": [
                {"status": state, "reason": reason, "count": count}
                for (state, reason), count in sorted(intakes.items())
            ],
            "prospectively_enrolled": len(entries),
            "verified_settled": len(settled),
            "distinct_settled_markets": len({row["ticker"] for row in settled}),
            "pending_outcomes": len(entries) - len(settled),
            "pending_label_problems": dict(sorted(Counter(
                row["last_label_problem"] or "not_yet_settled"
                for row in entries if row["state"] != "SETTLED"
            ).items())),
            "hypothetical_settlement_wins": sum(row["payout"] == 1.0 for row in settled),
            "hypothetical_settlement_losses": sum(row["payout"] == 0.0 for row in settled),
            "hypothetical_net_pnl": round(total_net, 6),
            "hypothetical_net_per_verified_entry": (
                round(total_net / len(settled), 6) if settled else None
            ),
        }
    meta = {
        key: int(value)
        for key, value in output.execute("SELECT key, value FROM shadow_meta")
    }
    return {
        "mode": "PROSPECTIVE_VIRTUAL_SETTLEMENT_SHADOW_ONLY",
        "no_exchange_orders": True,
        "no_automated_model_promotion": True,
        "paper_cursor": meta,
        "strategies": strategies,
        "limitations": [
            "A fresh independent observer starts AFTER all existing paper decisions; no retroactive enrollment.",
            "Original paper ENTRYs assume unconfirmed ask-side fills and entry fees/slippage.",
            "Settlements are independently matched after market close, never used as features.",
            "No executable exchange orders or fills; P&L is a virtual one-contract payout calculation.",
            "Pending/unmatched settlement labels have no imputed wins, losses, or P&L.",
            "Strategies may share markets; no aggregate portfolio profit can be inferred.",
            "Prospective observation and a large, untouched market cohort are required before edge claims.",
        ],
    }


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Settlement shadow observer is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Virtual settlement-shadow only; zero orders")
    parser.add_argument("--research-db", help="Collector SQLite database (read-only)")
    parser.add_argument("--paper-db", help="Existing paper simulator ledger (read-only)")
    parser.add_argument("--outcomes-db", help="Settled outcomes SQLite database (read-only)")
    parser.add_argument("--ledger-db", default="data/settlement_shadow/shadow.db")
    parser.add_argument("--loop", action="store_true", help="Loop prospectively (requires approval)")
    parser.add_argument("--status", action="store_true", help="Read current shadow ledger")
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("Poll interval must be at least one second")

    if args.status:
        if not Path(args.ledger_db).is_file():
            print("No shadow ledger exists")
            return
        with closing(_readonly(args.ledger_db)) as ledger:
            print(json.dumps(report(ledger), sort_keys=True, indent=2))
        return
    if not all((args.research_db, args.paper_db, args.outcomes_db)):
        parser.error("--research-db, --paper-db and --outcomes-db are required")
    source_paths = {Path(path).expanduser().resolve() for path in
                    (args.research_db, args.paper_db, args.outcomes_db)}
    dest = Path(args.ledger_db).expanduser().resolve()
    if dest in source_paths:
        parser.error("Shadow output must be a separate database from all read-only inputs")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with exclusive_lock(dest.with_suffix(".lock")):
        with closing(_readonly(args.research_db)) as research:
            with closing(_readonly(args.paper_db)) as paper:
                with closing(_readonly(args.outcomes_db)) as outcomes:
                    with closing(initialize_ledger(args.ledger_db, paper, now_ms=int(time.time() * 1000))) as db:
                        while True:
                            progress = tick(research, paper, outcomes, db)
                            snapshot = report(db)
                            if not args.loop:
                                print(json.dumps(snapshot, sort_keys=True, indent=2))
                                break
                            if progress["new_paper_decisions"]:
                                LOG.info("Observed %s decisions; cursor %s",
                                         progress["new_paper_decisions"], progress["last_decision_rowid"])
                            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
