"""Offline tests for verified binary settlement counterfactual; not actual trades."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.settlement_hold_comparison import diagnose

CLOSE = datetime(2026, 10, 19, 23, 0, tzinfo=timezone.utc)
CLOSE_MS = int(CLOSE.timestamp() * 1000)
ENTRY_MS = CLOSE_MS - 270000


def _fixture(tmp_path):
    research = tmp_path / "research.db"
    paper = tmp_path / "paper.db"
    outcomes = tmp_path / "outcomes.db"
    with sqlite3.connect(research) as db:
        db.executescript("""
            CREATE TABLE research_quotes (
                id INTEGER PRIMARY KEY, ticker TEXT, source TEXT,
                received_ms INTEGER, valid INTEGER, yes_ask REAL, no_ask REAL
            );
            CREATE TABLE research_markets (ticker TEXT PRIMARY KEY, close_ms INTEGER);
        """)
        db.executemany("INSERT INTO research_markets VALUES (?, ?)", [
            ("A", CLOSE_MS), ("B", CLOSE_MS), ("C", CLOSE_MS),
        ])
        db.executemany("INSERT INTO research_quotes VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (10, "A", "kalshi_rest", ENTRY_MS, 1, .40, .62),
            (20, "B", "kalshi_rest", ENTRY_MS, 1, .40, .62),
            (30, "C", "kalshi_rest", ENTRY_MS, 1, .36, .66),
        ])
    with sqlite3.connect(paper) as db:
        db.executescript("""
            CREATE TABLE paper_trades (
                id INTEGER PRIMARY KEY, ticker TEXT, strategy TEXT,
                quote_source TEXT, side TEXT, opened_quote_id INTEGER,
                opened_ms INTEGER, close_ms INTEGER, entry_ask REAL,
                entry_fee REAL, entry_slippage REAL, pnl_per_contract REAL,
                status TEXT
            );
        """)
        db.executemany("INSERT INTO paper_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            (1, "A", "price_rule", "kalshi_rest", "up", 10,
             ENTRY_MS, CLOSE_MS, .40, .02, .01, -.11, "CLOSED"),
            (2, "B", "momentum_rule", "kalshi_rest", "down", 20,
             ENTRY_MS, CLOSE_MS, .62, .02, .01, None, "CENSORED"),
            (3, "C", "momentum_rule", "kalshi_rest", "down", 30,
             ENTRY_MS, CLOSE_MS, .66, .02, .01, None, "OPEN"),
        ])
    with sqlite3.connect(outcomes) as db:
        db.executescript("""
            CREATE TABLE settled_outcomes (
                ticker TEXT PRIMARY KEY, result TEXT, settlement_value REAL,
                close_time_utc TEXT, pulled_at_utc TEXT
            );
        """)
        db.executemany("INSERT INTO settled_outcomes VALUES (?, ?, ?, ?, ?)", [
            ("A", "yes", 1.0, CLOSE.isoformat(), (CLOSE + timedelta(minutes=3)).isoformat()),
            ("B", "yes", 1.0, CLOSE.isoformat(), (CLOSE + timedelta(minutes=3)).isoformat()),
            ("C", "no", 0.0, CLOSE.isoformat(), (CLOSE + timedelta(minutes=3)).isoformat()),
        ])
    return str(research), str(paper), str(outcomes)


def test_binary_wins_losses_and_paired_comparison_no_writes(tmp_path):
    research, paper, outcomes = _fixture(tmp_path)
    before = [(tmp_path / name).read_bytes() for name in ("research.db", "paper.db", "outcomes.db")]
    result = diagnose(research, paper, outcomes)
    assert result["actual_orders"] is False
    assert result["does_not_use_outcome_to_select_entry"] is True

    price = result["strategies"]["price_rule"]
    assert price["validated_settlement_entries"] == 1
    assert price["settlement_winning_entries"] == 1
    assert price["hypothetical_hold_net_total"] == pytest.approx(.57)
    assert price["paired_original_closed_pnl"] == pytest.approx(-.11)
    assert price["paired_hold_minus_original"] == pytest.approx(.68)
    assert price["paired_distinct_markets"] == 1

    momentum = result["strategies"]["momentum_rule"]
    assert momentum["recorded_entries"] == 2
    assert momentum["validated_distinct_markets"] == 2
    assert momentum["settlement_winning_entries"] == 1
    assert momentum["settlement_losing_entries"] == 1
    assert momentum["hypothetical_hold_net_total"] == pytest.approx(-.34)
    assert momentum["paired_original_closed_count"] == 0
    assert momentum["original_status_counts"] == {"CENSORED": 1, "OPEN": 1}
    after = [(tmp_path / name).read_bytes() for name in ("research.db", "paper.db", "outcomes.db")]
    assert after == before


def test_missing_and_inconsistent_settlements_are_excluded(tmp_path):
    research, paper, outcomes = _fixture(tmp_path)
    with sqlite3.connect(outcomes) as db:
        db.execute("DELETE FROM settled_outcomes WHERE ticker='A'")
        db.execute("UPDATE settled_outcomes SET settlement_value=NULL WHERE ticker='B'")
        db.execute("UPDATE settled_outcomes SET settlement_value=1 WHERE ticker='C'")
    result = diagnose(research, paper, outcomes)["strategies"]
    assert result["price_rule"]["excluded_reasons"] == {"no_stored_settlement": 1}
    assert result["price_rule"]["hypothetical_hold_net_total"] == 0
    assert result["momentum_rule"]["validated_settlement_entries"] == 0
    assert result["momentum_rule"]["excluded_reasons"] == {
        "missing_binary_settlement_value": 1,
        "settlement_result_value_conflict": 1,
    }


def test_settlement_timestamp_guards_and_entry_provenance(tmp_path):
    research, paper, outcomes = _fixture(tmp_path)
    with sqlite3.connect(outcomes) as db:
        db.execute(
            "UPDATE settled_outcomes SET pulled_at_utc=? WHERE ticker='A'",
            ((CLOSE - timedelta(seconds=30)).isoformat(),),
        )
        db.execute(
            "UPDATE settled_outcomes SET close_time_utc=? WHERE ticker='B'",
            ((CLOSE + timedelta(minutes=5)).isoformat(),),
        )
    with sqlite3.connect(paper) as db:
        db.execute("UPDATE paper_trades SET entry_ask=.55 WHERE id=3")
    result = diagnose(research, paper, outcomes)["strategies"]
    assert result["price_rule"]["excluded_reasons"] == {"settlement_reported_before_close": 1}
    assert result["momentum_rule"]["excluded_reasons"] == {
        "settlement_close_time_disagrees": 1,
        "invalid_entry_quote_or_cost": 1,
    }
    assert result["momentum_rule"]["validated_settlement_entries"] == 0


def test_nonbinary_outcome_is_rejected_without_inventing_dollar_payoff(tmp_path):
    research, paper, outcomes = _fixture(tmp_path)
    with sqlite3.connect(outcomes) as db:
        db.execute("UPDATE settled_outcomes SET settlement_value=0.5 WHERE ticker='A'")
    result = diagnose(research, paper, outcomes)["strategies"]["price_rule"]
    assert result["validated_settlement_entries"] == 0
    assert result["excluded_reasons"] == {"nonbinary_settlement_value": 1}
    assert result["hypothetical_hold_net_per_settled_entry"] is None
