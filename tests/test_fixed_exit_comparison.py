"""Offline tests for retrospective fixed-exit sensitivity, never actual fills."""
from __future__ import annotations

import sqlite3

import pytest

from src.fixed_exit_comparison import diagnose


def _fixture(tmp_path):
    research_file = tmp_path / "research.db"
    paper_file = tmp_path / "paper.db"
    with sqlite3.connect(research_file) as db:
        db.executescript("""
            CREATE TABLE research_quotes(
                id INTEGER PRIMARY KEY, ticker TEXT, source TEXT, received_ms INTEGER,
                valid INTEGER, yes_bid REAL, yes_ask REAL, no_bid REAL, no_ask REAL,
                yes_depth REAL, no_depth REAL
            );
        """)
        rows = []
        for i, timestamp in enumerate(range(100000, 395001, 5000), start=100):
            # Entry: 30c ask, 29c bid. A rapid 5-second dip stops the original.
            # Later fixed exit outcomes are different, all using observed quotes.
            if timestamp == 100000:
                bid = .29
            elif timestamp == 105000:
                bid = .25
            elif timestamp < 355000:
                bid = .34
            else:
                bid = .60
            ask = bid + .01
            rows.append((i, "A", "kalshi_rest", timestamp, 1, bid, ask,
                         1 - ask, 1 - bid, 10, 10))
        db.executemany("INSERT INTO research_quotes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    with sqlite3.connect(paper_file) as db:
        db.executescript("""
            CREATE TABLE paper_trades(
                id INTEGER PRIMARY KEY, ticker TEXT, strategy TEXT, quote_source TEXT,
                side TEXT, opened_ms INTEGER, opened_quote_id INTEGER, close_ms INTEGER,
                entry_ask REAL, entry_fee REAL, entry_slippage REAL, status TEXT,
                last_quote_ms INTEGER, pnl_per_contract REAL
            );
            INSERT INTO paper_trades VALUES
                (1, 'A', 'price_rule', 'kalshi_rest', 'up', 100000, 100, 400000,
                 .30, .02, .01, 'CLOSED', 105000, -.11);
        """)
    return str(research_file), str(paper_file)


def test_fixed_exits_compare_on_same_original_entry_without_modifications(tmp_path):
    source, paper = _fixture(tmp_path)
    before = ((tmp_path / "research.db").read_bytes(), (tmp_path / "paper.db").read_bytes())
    result = diagnose(source, paper, now_ms=500000)
    assert result["no_exchange_orders"] is True
    strategies = result["strategies"]["price_rule"]
    hold = strategies["hold_60_seconds"]
    late = strategies["hold_to_preclose"]
    assert hold["candidate_status_counts"] == {"CLOSED": 1}
    assert late["candidate_status_counts"] == {"CLOSED": 1}
    assert hold["candidate_completed_virtual_pnl"] == pytest.approx(-.02)
    assert late["candidate_completed_virtual_pnl"] == pytest.approx(.24)
    assert hold["paired_original_virtual_pnl"] == pytest.approx(-.11)
    assert hold["paired_candidate_minus_original"] == pytest.approx(.09)
    assert late["paired_candidate_minus_original"] == pytest.approx(.35)
    assert late["distinct_paired_markets"] == 1
    after = ((tmp_path / "research.db").read_bytes(), (tmp_path / "paper.db").read_bytes())
    assert before == after


def test_gap_before_target_censors_even_if_future_exit_would_be_profitable(tmp_path):
    source, paper = _fixture(tmp_path)
    with sqlite3.connect(source) as db:
        db.execute(
            "DELETE FROM research_quotes WHERE received_ms>110000 AND received_ms<355000"
        )
    result = diagnose(source, paper, now_ms=500000)["strategies"]["price_rule"]
    for policy in ("hold_60_seconds", "hold_to_preclose"):
        assert result[policy]["candidate_status_counts"] == {"CENSORED": 1}
        assert result[policy]["candidate_completed_virtual_pnl"] == 0
        assert result[policy]["paired_candidate_minus_original"] == 0
        assert result[policy]["candidate_censor_or_pending_reasons"] == {
            "gap_exceeds_30_seconds": 1
        }


def test_original_censor_boundary_blocks_replay_of_unobserved_prices(tmp_path):
    source, paper = _fixture(tmp_path)
    with sqlite3.connect(paper) as db:
        db.execute(
            "UPDATE paper_trades SET status='CENSORED', last_quote_ms=105000, "
            "pnl_per_contract=NULL WHERE id=1"
        )
    result = diagnose(source, paper, now_ms=500000)["strategies"]["price_rule"]
    assert result["hold_60_seconds"]["candidate_status_counts"] == {"CENSORED": 1}
    assert result["hold_to_preclose"]["candidate_status_counts"] == {"CENSORED": 1}
    assert result["hold_to_preclose"]["original_closed_candidate_closed_pairs"] == 0


def test_open_position_is_pending_and_not_assigned_pnl(tmp_path):
    source, paper = _fixture(tmp_path)
    with sqlite3.connect(paper) as db:
        db.execute(
            "UPDATE paper_trades SET status='OPEN', last_quote_ms=115000, "
            "pnl_per_contract=NULL WHERE id=1"
        )
    result = diagnose(source, paper, now_ms=120000)["strategies"]["price_rule"]
    assert result["hold_60_seconds"]["candidate_status_counts"] == {"PENDING": 1}
    assert result["hold_60_seconds"]["candidate_completed_virtual_pnl"] == 0


def test_source_quote_mismatch_is_unverifiable_not_filled(tmp_path):
    source, paper = _fixture(tmp_path)
    with sqlite3.connect(paper) as db:
        db.execute("UPDATE paper_trades SET entry_ask=.50 WHERE id=1")
    result = diagnose(source, paper, now_ms=500000)["strategies"]["price_rule"]
    assert result["hold_60_seconds"]["candidate_status_counts"] == {"UNVERIFIABLE": 1}
    assert result["hold_60_seconds"]["original_closed_candidate_closed_pairs"] == 0
