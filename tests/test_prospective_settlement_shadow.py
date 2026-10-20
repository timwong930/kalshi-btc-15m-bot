"""Offline tests for forward-only settlement observation, never exchange orders."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from src.prospective_settlement_shadow import initialize_ledger, report, tick

T = 1792450000000
CLOSE = T + 100000


def iso_ms(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def setup(tmp_path):
    research = tmp_path / "research.db"
    paper = tmp_path / "paper.db"
    outcome = tmp_path / "outcomes.db"
    shadow = tmp_path / "shadow.db"
    with sqlite3.connect(research) as db:
        db.executescript("""
            CREATE TABLE research_markets(ticker TEXT PRIMARY KEY, close_ms INTEGER);
            CREATE TABLE research_quotes(
                id INTEGER PRIMARY KEY, ticker TEXT, source TEXT,
                received_ms INTEGER, valid INTEGER, yes_ask REAL, no_ask REAL
            );
        """)
        db.executemany("INSERT INTO research_markets VALUES (?, ?)", [("A", CLOSE), ("B", CLOSE)])
        db.executemany("INSERT INTO research_quotes VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (1, "A", "kalshi_rest", T - 20000, 1, .3, .7),
            (2, "A", "kalshi_rest", T + 1000, 1, .3, .7),
            (3, "B", "kalshi_rest", T + 1000, 1, .3, .7),
        ])
    with sqlite3.connect(paper) as db:
        db.executescript("""
            CREATE TABLE paper_decisions (
                ticker TEXT NOT NULL, strategy TEXT NOT NULL, event_ms INTEGER,
                quote_id INTEGER, decision TEXT, reason TEXT,
                PRIMARY KEY(ticker, strategy)
            );
            CREATE TABLE paper_trades (
                ticker TEXT, strategy TEXT, opened_quote_id INTEGER,
                opened_ms INTEGER, close_ms INTEGER, quote_source TEXT,
                side TEXT, entry_ask REAL, entry_fee REAL, entry_slippage REAL,
                PRIMARY KEY(ticker, strategy)
            );
        """)
        db.execute("INSERT INTO paper_decisions VALUES (?, ?, ?, ?, ?, ?)",
                   ("A", "price_rule", T - 20000, 1, "ENTRY", "fixed_price_rule"))
        db.execute("INSERT INTO paper_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   ("A", "price_rule", 1, T - 20000, CLOSE, "kalshi_rest", "up", .3, .02, .01))
    with sqlite3.connect(outcome) as db:
        db.executescript("""
            CREATE TABLE settled_outcomes (
                ticker TEXT PRIMARY KEY, result TEXT, settlement_value REAL,
                close_time_utc TEXT, pulled_at_utc TEXT
            );
        """)
    return research, paper, outcome, shadow


def later(paper):
    with sqlite3.connect(paper) as db:
        db.executemany("INSERT INTO paper_decisions VALUES (?, ?, ?, ?, ?, ?)", [
            ("A", "momentum_rule", T + 1000, 2, "ENTRY", "fixed_momentum_rule"),
            ("B", "outcome_logistic", T + 1000, 3, "ENTRY", "shadow_outcome_edge"),
            ("B", "reversal_logistic", T + 1000, 3, "SKIP", "below_threshold"),
        ])
        db.executemany("INSERT INTO paper_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            ("A", "momentum_rule", 2, T + 1000, CLOSE, "kalshi_rest", "up", .3, .02, .01),
            ("B", "outcome_logistic", 3, T + 1000, CLOSE, "kalshi_rest", "down", .7, .02, .01),
        ])


def settlements(outcome, pulled_at):
    with sqlite3.connect(outcome) as db:
        db.executemany("INSERT INTO settled_outcomes VALUES (?, ?, ?, ?, ?)", [
            ("A", "yes", 1.0, iso_ms(CLOSE), iso_ms(pulled_at)),
            ("B", "yes", 1.0, iso_ms(CLOSE), iso_ms(pulled_at)),
        ])


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def test_forward_only_and_post_close_labels(tmp_path):
    research, paper, outcomes, shadow = setup(tmp_path)
    src, p, o = connect(research), connect(paper), connect(outcomes)
    ledger = initialize_ledger(str(shadow), p, now_ms=T)
    try:
        assert tick(src, p, o, ledger, now_ms=T)["new_paper_decisions"] == 0
        later(paper)
        assert tick(src, p, o, ledger, now_ms=T + 2000)["new_paper_decisions"] == 3
        assert tick(src, p, o, ledger, now_ms=T + 2000)["new_paper_decisions"] == 0
        assert "price_rule" not in report(ledger)["strategies"]
        assert report(ledger)["strategies"]["reversal_logistic"]["prospectively_enrolled"] == 0
        settlements(outcomes, CLOSE + 5000)
        tick(src, p, o, ledger, now_ms=T + 30000)
        assert report(ledger)["strategies"]["momentum_rule"]["verified_settled"] == 0
        tick(src, p, o, ledger, now_ms=CLOSE + 6000)
        results = report(ledger)["strategies"]
        assert results["momentum_rule"]["hypothetical_net_pnl"] == pytest.approx(.67)
        assert results["outcome_logistic"]["hypothetical_net_pnl"] == pytest.approx(-.73)
        tick(src, p, o, ledger, now_ms=CLOSE + 7000)
        assert report(ledger)["strategies"]["momentum_rule"]["verified_settled"] == 1
    finally:
        ledger.close()
        src.close()
        p.close()
        o.close()


def test_stale_entry_is_rejected(tmp_path):
    research, paper, outcomes, shadow = setup(tmp_path)
    src, p, o = connect(research), connect(paper), connect(outcomes)
    ledger = initialize_ledger(str(shadow), p, now_ms=T)
    try:
        later(paper)
        tick(src, p, o, ledger, now_ms=T + 40000)
        result = report(ledger)["strategies"]["momentum_rule"]
        assert result["prospectively_enrolled"] == 0
        assert result["paper_intake_reasons"] == [{
            "status": "REJECTED", "reason": "decision_not_observed_promptly", "count": 1,
        }]
    finally:
        ledger.close()
        src.close()
        p.close()
        o.close()


def test_pending_and_future_dated_settlement_are_not_fabricated(tmp_path):
    research, paper, outcomes, shadow = setup(tmp_path)
    src, p, o = connect(research), connect(paper), connect(outcomes)
    ledger = initialize_ledger(str(shadow), p, now_ms=T)
    try:
        later(paper)
        tick(src, p, o, ledger, now_ms=T + 2000)
        tick(src, p, o, ledger, now_ms=CLOSE + 6000)
        result = report(ledger)["strategies"]["momentum_rule"]
        assert result["verified_settled"] == 0
        assert result["pending_label_problems"] == {"no_stored_settlement": 1}
        settlements(outcomes, CLOSE + 30000)
        tick(src, p, o, ledger, now_ms=CLOSE + 10000)
        result = report(ledger)["strategies"]["momentum_rule"]
        assert result["pending_label_problems"] == {"settlement_record_in_future": 1}
        tick(src, p, o, ledger, now_ms=CLOSE + 30001)
        assert report(ledger)["strategies"]["momentum_rule"]["verified_settled"] == 1
    finally:
        ledger.close()
        src.close()
        p.close()
        o.close()


def test_restart_preserves_cursor_and_entries(tmp_path):
    research, paper, outcomes, shadow = setup(tmp_path)
    src, p, o = connect(research), connect(paper), connect(outcomes)
    ledger = initialize_ledger(str(shadow), p, now_ms=T)
    try:
        later(paper)
        tick(src, p, o, ledger, now_ms=T + 2000)
        original = report(ledger)["paper_cursor"]["last_decision_rowid"]
    finally:
        ledger.close()
    resumed = initialize_ledger(str(shadow), p, now_ms=CLOSE + 5000)
    try:
        assert tick(src, p, o, resumed, now_ms=CLOSE + 5000)["new_paper_decisions"] == 0
        assert report(resumed)["paper_cursor"]["last_decision_rowid"] == original
        assert report(resumed)["strategies"]["momentum_rule"]["prospectively_enrolled"] == 1
    finally:
        resumed.close()
        src.close()
        p.close()
        o.close()


def test_bad_original_quote_does_not_enroll(tmp_path):
    research, paper, outcomes, shadow = setup(tmp_path)
    src, p, o = connect(research), connect(paper), connect(outcomes)
    ledger = initialize_ledger(str(shadow), p, now_ms=T)
    try:
        later(paper)
        with sqlite3.connect(research) as db:
            db.execute("UPDATE research_quotes SET yes_ask=.99 WHERE id=2")
        tick(src, p, o, ledger, now_ms=T + 2000)
        results = report(ledger)["strategies"]
        assert results["momentum_rule"]["paper_intake_reasons"][0]["reason"] == (
            "invalid_entry_quote_or_cost"
        )
        assert results["outcome_logistic"]["prospectively_enrolled"] == 1
    finally:
        ledger.close()
        src.close()
        p.close()
        o.close()
