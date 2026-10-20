"""Offline regression tests for entry-spread/stop feasibility (not actual fills)."""
from __future__ import annotations

import sqlite3

import pytest

from src.entry_stop_audit import diagnose


def _databases(tmp_path):
    research_path = tmp_path / "research.db"
    paper_path = tmp_path / "paper.db"
    with sqlite3.connect(research_path) as research:
        research.executescript("""
            CREATE TABLE research_quotes (
                id INTEGER PRIMARY KEY, ticker TEXT, source TEXT,
                yes_bid REAL, yes_ask REAL, no_bid REAL, no_ask REAL
            );
            INSERT INTO research_quotes VALUES
                (100, 'A', 'kalshi_rest', 0.25, 0.30, 0.70, 0.75),
                (101, 'B', 'kalshi_rest', 0.28, 0.30, 0.70, 0.72),
                (102, 'C', 'kalshi_rest', 0.23, 0.30, 0.70, 0.77),
                (103, 'D', 'kalshi_rest', 0.22, 0.30, 0.70, 0.78);
        """)
    with sqlite3.connect(paper_path) as paper:
        paper.executescript("""
            CREATE TABLE paper_trades (
                id INTEGER PRIMARY KEY, ticker TEXT, strategy TEXT,
                quote_source TEXT, side TEXT, opened_quote_id INTEGER,
                entry_ask REAL, entry_fee REAL, entry_slippage REAL,
                status TEXT, reason TEXT
            );
            INSERT INTO paper_trades VALUES
                (1, 'A', 'price_rule', 'kalshi_rest', 'up', 100,
                 0.30, 0.02, 0.01, 'CLOSED', 'stop_loss'),
                (2, 'B', 'outcome_logistic', 'kalshi_rest', 'up', 101,
                 0.30, 0.02, 0.01, 'CENSORED', 'invalid_exit_book_or_depth'),
                (3, 'C', 'momentum_rule', 'kalshi_rest', 'up', 102,
                 0.30, 0.02, 0.01, 'OPEN', NULL),
                (4, 'NOT-D', 'momentum_rule', 'kalshi_rest', 'up', 103,
                 0.30, 0.02, 0.01, 'OPEN', NULL);
        """)
    return str(research_path), str(paper_path)


def test_entry_spread_can_already_trigger_the_net_stop(tmp_path):
    research, paper = _databases(tmp_path)
    results = diagnose(research, paper)
    assert results["not_real_fills"] is True
    price = results["strategies"]["price_rule"]
    assert price["recorded_entries"] == 1
    assert price["matched_entry_quotes"] == 1
    assert price["entry_spread_dollars"]["mean"] == pytest.approx(0.05)
    assert price["initial_net_mark_at_unchanged_bid_dollars"]["mean"] == pytest.approx(-0.11)
    assert price["bid_rise_needed_from_entry_bid_to_break_even"]["mean"] == pytest.approx(0.11)
    assert price["bid_rise_needed_from_entry_bid_to_net_take_profit"]["mean"] == pytest.approx(0.21)
    assert price["already_meets_net_stop_if_entry_bid_unchanged"] == 1
    assert price["already_meets_net_stop_closed_exit_reasons"] == {"stop_loss": 1}

    outcome = results["strategies"]["outcome_logistic"]
    assert outcome["already_meets_net_stop_if_entry_bid_unchanged"] == 0
    assert outcome["initial_net_mark_at_unchanged_bid_dollars"]["mean"] == pytest.approx(-0.08)
    assert outcome["status_counts"] == {"CENSORED": 1}


def test_unmatched_quotes_reported_and_censored_positions_not_fake_losses(tmp_path):
    research, paper = _databases(tmp_path)
    before = (tmp_path / "research.db").read_bytes(), (tmp_path / "paper.db").read_bytes()
    results = diagnose(research, paper)
    momentum = results["strategies"]["momentum_rule"]
    assert momentum["recorded_entries"] == 2
    assert momentum["matched_entry_quotes"] == 1
    assert momentum["unmatched_entry_quotes"] == 1
    assert momentum["unmatched_reasons"] == {"ticker_or_source_mismatch": 1}
    assert momentum["status_counts"] == {"OPEN": 2}
    assert "profit" not in momentum
    assert momentum["already_meets_net_stop_if_entry_bid_unchanged"] == 1
    after = (tmp_path / "research.db").read_bytes(), (tmp_path / "paper.db").read_bytes()
    assert after == before


def test_invalid_entry_price_cannot_be_inferred_from_other_quote(tmp_path):
    research, paper = _databases(tmp_path)
    with sqlite3.connect(paper) as db:
        db.execute("UPDATE paper_trades SET entry_ask=0.40 WHERE id=1")
        db.execute("UPDATE paper_trades SET opened_quote_id=98765 WHERE id=2")
    results = diagnose(research, paper)["strategies"]
    assert results["price_rule"]["matched_entry_quotes"] == 0
    assert results["price_rule"]["already_meets_net_stop_rate"] is None
    assert results["price_rule"]["unmatched_reasons"] == {"invalid_or_mismatched_entry_quote": 1}
    assert results["outcome_logistic"]["unmatched_reasons"] == {"missing_source_quote": 1}
