"""Self-contained regression tests for the read-only research diagnostic."""
from __future__ import annotations

import sqlite3

from src.research_diagnostics import diagnose


def _fixture(tmp_path):
    source_path = tmp_path / "research.db"
    db = sqlite3.connect(source_path)
    db.executescript("""
        CREATE TABLE research_markets(ticker TEXT PRIMARY KEY, close_ms INTEGER,
            first_seen_ms INTEGER);
        CREATE TABLE research_quotes(id INTEGER PRIMARY KEY, ticker TEXT,
            source TEXT, received_ms INTEGER, valid INTEGER, yes_bid REAL,
            yes_ask REAL, no_bid REAL, no_ask REAL, yes_depth REAL, no_depth REAL);
    """)
    db.executemany("INSERT INTO research_markets VALUES (?, ?, ?)", [
        ("A", 310000, 100), ("B", 400000, 90000), ("C", 500000, 400000),
    ])
    good = (.25, .30, .70, .75, 20, 20)
    bad = (.25, .40, .60, .75, 20, 20)
    db.executemany("INSERT INTO research_quotes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
        (1, "A", "kalshi_rest", 1000, 1, *good),
        (2, "A", "kalshi_rest", 15000, 1, *good),
        (3, "A", "kalshi_rest", 50000, 1, *good),
        (4, "B", "kalshi_rest", 100000, 1, *bad),
    ])
    db.commit()
    db.close()
    ledger_path = tmp_path / "paper.db"
    paper = sqlite3.connect(ledger_path)
    paper.executescript("""
        CREATE TABLE paper_meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE paper_decisions(strategy TEXT, decision TEXT, reason TEXT);
        CREATE TABLE paper_trades(strategy TEXT, status TEXT, reason TEXT);
        INSERT INTO paper_meta VALUES('last_quote_id', '4');
        INSERT INTO paper_decisions VALUES('reversal_logistic', 'SKIP', 'reversal_probability_below_threshold');
        INSERT INTO paper_trades VALUES('momentum_rule', 'CENSORED', 'missing_or_delayed_exit_quotes');
    """)
    paper.commit()
    paper.close()
    return str(source_path), str(ledger_path)


def test_separates_cross_market_gaps_and_static_book_filter(tmp_path):
    source, ledger = _fixture(tmp_path)
    result = diagnose(source, ledger, now_ms=800000)
    research = result["research"]
    assert research["global_consecutive_gaps"]["over_30s"] == 2
    assert research["global_consecutive_gaps"]["different_ticker"] == 1
    assert research["global_consecutive_gaps"]["same_ticker_same_source"] == 1
    assert research["within_same_ticker_source_gaps"]["over_30s"] == 1
    assert research["quotes"]["valid_flag_true"] == 4
    assert research["quotes"]["passes_static_book_filter"] == 3
    coverage = research["entry_window_coverage"]
    assert coverage["completed_windows"] == 3
    assert coverage["has_static_book_eligible_quote"] == 1
    assert coverage["quotes_present_none_book_eligible"] == 1
    assert coverage["no_quote_in_entry_window"] == 1
    assert coverage["first_discovered_after_window_open"] == 1
    assert result["paper"]["last_quote_id"] == 4
    assert result["paper"]["decision_reasons"][0]["count"] == 1
    assert result["paper"]["trade_status_reasons"][0]["reason"] == "missing_or_delayed_exit_quotes"


def test_can_run_without_paper_ledger(tmp_path):
    source, _ = _fixture(tmp_path)
    result = diagnose(source, now_ms=800000)
    assert result["paper"] is None
    assert result["read_only"] is True
