"""Regression tests for quote-path analysis of virtual trades, never real fills."""
from __future__ import annotations

import sqlite3

import pytest

from src.paper_trade_path_audit import diagnose


def fixture_dbs(tmp_path):
    research_path = tmp_path / "research.db"
    paper_path = tmp_path / "paper.db"
    with sqlite3.connect(research_path) as db:
        db.executescript("""
            CREATE TABLE research_quotes (
                id INTEGER PRIMARY KEY, ticker TEXT, source TEXT, received_ms INTEGER,
                yes_bid REAL, yes_ask REAL, no_bid REAL, no_ask REAL,
                yes_depth REAL, no_depth REAL, valid INTEGER
            );
            CREATE TABLE research_prices (
                id INTEGER PRIMARY KEY, received_ms INTEGER, source TEXT, price REAL
            );
            INSERT INTO research_quotes VALUES
                (10, 'A', 'kalshi_rest', 100000, .29, .30, .70, .71, 10, 10, 1),
                (11, 'A', 'kalshi_rest', 101500, .265, .275, .725, .735, 10, 10, 1),
                (12, 'A', 'kalshi_rest', 104500, .25, .26, .74, .75, 10, 10, 1),
                (20, 'B', 'kalshi_rest', 200000, .49, .50, .50, .51, 10, 10, 1),
                (21, 'B', 'kalshi_rest', 220000, .68, .69, .31, .32, 10, 10, 1);
            INSERT INTO research_prices VALUES
                (1, 99900, 'coinbase', 100000),
                (2, 104400, 'coinbase', 100050),
                (3, 199500, 'coinbase', 100000),
                (4, 219800, 'coinbase', 100200);
        """)
    with sqlite3.connect(paper_path) as db:
        db.executescript("""
            CREATE TABLE paper_trades (
                id INTEGER PRIMARY KEY, ticker TEXT, strategy TEXT,
                quote_source TEXT, side TEXT, opened_quote_id INTEGER,
                exit_quote_id INTEGER, opened_ms INTEGER, exit_ms INTEGER,
                entry_ask REAL, exit_bid REAL, entry_fee REAL, entry_slippage REAL,
                exit_fee REAL, exit_slippage REAL, pnl_per_contract REAL,
                status TEXT, reason TEXT
            );
            INSERT INTO paper_trades VALUES
                (1, 'A', 'momentum_rule', 'kalshi_rest', 'up', 10, 12,
                 100000, 104500, .30, .25, .02, .01, .02, .01, -.11,
                 'CLOSED', 'stop_loss'),
                (2, 'B', 'outcome_logistic', 'kalshi_rest', 'up', 20, 21,
                 200000, 220000, .50, .68, .02, .01, .02, .01, .12,
                 'CLOSED', 'take_profit'),
                (3, 'B', 'price_rule', 'kalshi_rest', 'up', 20, NULL,
                 200000, NULL, .50, NULL, .02, .01, NULL, NULL, NULL,
                 'CENSORED', 'invalid_exit_book_or_depth');
        """)
    return str(research_path), str(paper_path)


def test_uses_quote_ids_and_decomposes_movement_and_spread(tmp_path):
    research, paper = fixture_dbs(tmp_path)
    before = (tmp_path / "research.db").read_bytes(), (tmp_path / "paper.db").read_bytes()
    data = diagnose(research, paper, include_trades=True)
    assert data["never_actual_fills"] is True
    momentum = data["strategies"]["momentum_rule"]
    assert momentum["analyzed_closed_trades"] == 1
    assert momentum["stops_with_exit_in_15_seconds"] == 1
    assert momentum["stops_at_first_later_structurally_valid_quote"] == 0
    assert momentum["coinbase_valid_pairs"] == 1
    path = momentum["closed_trade_paths"][0]
    assert path["time_to_exit_seconds"] == 4.5
    assert path["observed_later_quote_count"] == 2
    assert path["entry_spread"] == pytest.approx(.01)
    assert path["exit_bid_change_from_entry_bid"] == pytest.approx(-.04)
    assert path["gross_quote_movement"] == pytest.approx(-.05)
    assert path["recorded_round_trip_cost"] == pytest.approx(.06)
    assert path["recorded_virtual_net_pnl"] == pytest.approx(-.11)
    assert path["bid_drift_to_reach_net_stop_from_entry_bid"] == pytest.approx(-.03)
    assert path["first_later_structurally_valid_quote_at_net_stop"] is False
    assert path["coinbase_side_aligned_move_bps"] == pytest.approx(5)

    logistic = data["strategies"]["outcome_logistic"]
    assert logistic["closed_trade_paths"][0]["gross_quote_movement"] == pytest.approx(.18)
    assert logistic["closed_trade_paths"][0]["recorded_virtual_net_pnl"] == pytest.approx(.12)
    assert data["strategies"]["price_rule"]["trade_status_counts"] == {"CENSORED": 1}
    assert data["strategies"]["price_rule"]["analyzed_closed_trades"] == 0
    after = (tmp_path / "research.db").read_bytes(), (tmp_path / "paper.db").read_bytes()
    assert after == before


def test_first_eligible_later_quote_can_reach_stop(tmp_path):
    research, paper = fixture_dbs(tmp_path)
    with sqlite3.connect(research) as db:
        db.execute(
            "UPDATE research_quotes SET yes_bid=.25, yes_ask=.26, no_bid=.74, no_ask=.75 WHERE id=11"
        )
    with sqlite3.connect(paper) as db:
        db.execute(
            "UPDATE paper_trades SET exit_quote_id=11, exit_ms=101500 WHERE id=1"
        )
    data = diagnose(research, paper)
    momentum = data["strategies"]["momentum_rule"]
    assert momentum["analyzed_closed_trades"] == 1
    assert momentum["stops_at_first_later_structurally_valid_quote"] == 1
    assert momentum["stops_with_first_later_structurally_valid_quote_observed"] == 1


def test_mismatch_cannot_create_fabricated_trajectory(tmp_path):
    research, paper = fixture_dbs(tmp_path)
    with sqlite3.connect(paper) as db:
        db.execute("UPDATE paper_trades SET exit_bid=.99 WHERE id=1")
    data = diagnose(research, paper)["strategies"]
    assert data["momentum_rule"]["analyzed_closed_trades"] == 0
    assert data["momentum_rule"]["unmatched_reasons"] == {
        "entry_exit_price_or_pnl_mismatch": 1
    }
    assert data["momentum_rule"]["stops_with_exit_in_15_seconds"] == 0


def test_missing_coinbase_is_explicit_not_future_filled(tmp_path):
    research, paper = fixture_dbs(tmp_path)
    with sqlite3.connect(research) as db:
        db.execute("DELETE FROM research_prices")
    data = diagnose(research, paper)["strategies"]["momentum_rule"]
    assert data["analyzed_closed_trades"] == 1
    assert data["coinbase_valid_pairs"] == 0
    assert data["coinbase_missing_or_stale_pairs"] == 1
