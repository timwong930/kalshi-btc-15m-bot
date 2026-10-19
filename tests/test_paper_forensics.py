"""Offline regression tests for immutable virtual-trade cost attribution."""
from __future__ import annotations

import sqlite3

import pytest

from src.paper_forensics import diagnose


def make_ledger(tmp_path):
    path = tmp_path / "paper.db"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY, ticker TEXT, strategy TEXT, status TEXT, reason TEXT,
            entry_ask REAL, exit_bid REAL, entry_fee REAL, exit_fee REAL,
            entry_slippage REAL, exit_slippage REAL, pnl_per_contract REAL
        );
        CREATE TABLE paper_decisions(strategy TEXT, decision TEXT, reason TEXT);
        INSERT INTO paper_trades VALUES
            (1, 'MKT-A', 'price_rule', 'CLOSED', 'take_profit',
             0.30, 0.50, 0.02, 0.02, 0.01, 0.01, 0.14),
            (2, 'MKT-B', 'price_rule', 'CLOSED', 'stop_loss',
             0.30, 0.25, 0.02, 0.02, 0.01, 0.01, -0.11),
            (3, 'MKT-C', 'price_rule', 'CENSORED', 'invalid_exit_book_or_depth',
             0.31, NULL, 0.02, NULL, 0.01, NULL, NULL),
            (4, 'MKT-A', 'reversal_logistic', 'OPEN', NULL,
             0.31, NULL, 0.02, NULL, 0.01, NULL, NULL);
        INSERT INTO paper_decisions VALUES
            ('price_rule', 'ENTRY', 'fixed_price_rule'),
            ('price_rule', 'ENTRY', 'fixed_price_rule'),
            ('price_rule', 'ENTRY', 'fixed_price_rule'),
            ('reversal_logistic', 'SKIP', 'reversal_probability_below_threshold');
    """)
    connection.commit()
    connection.close()
    return path


def test_reconcile_and_cost_scenarios_without_mutating_source(tmp_path):
    path = make_ledger(tmp_path)
    before = path.read_bytes()
    result = diagnose(str(path))
    assert result["no_exchange_orders"]
    price = result["strategies"]["price_rule"]
    assert price["closed_trades"] == 2
    assert price["unique_closed_market_tickers"] == 2
    assert price["gross_quote_movement_dollars"] == pytest.approx(0.15)
    assert price["assumed_recorded_costs_dollars"] == pytest.approx(0.12)
    assert price["recorded_net_simulated_pnl_dollars"] == pytest.approx(0.03)
    assert price["mean_round_trip_recorded_cost"] == pytest.approx(0.06)
    assert price["exit_reasons"] == {"stop_loss": 1, "take_profit": 1}
    scenarios = {(x["fee_per_side"], x["slippage_per_side"]): x for x in price["cost_scenarios"]}
    assert scenarios[(0.0, 0.0)]["conditional_pnl_same_observed_exits"] == pytest.approx(0.15)
    assert scenarios[(0.02, 0.01)]["conditional_pnl_same_observed_exits"] == pytest.approx(0.03)
    assert scenarios[(0.03, 0.02)]["conditional_pnl_same_observed_exits"] == pytest.approx(-0.05)
    assert price["trade_status_counts"]["CENSORED"] == 1
    assert result["strategies"]["reversal_logistic"]["closed_trades"] == 0
    assert result["strategies"]["reversal_logistic"]["mean_round_trip_recorded_cost"] is None
    assert path.read_bytes() == before


def test_nonreconciling_ledger_refused(tmp_path):
    path = make_ledger(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE paper_trades SET pnl_per_contract=999 WHERE id=1")
    with pytest.raises(ValueError, match="does not reconcile"):
        diagnose(str(path))


def test_empty_ledger_has_zero_results(tmp_path):
    path = make_ledger(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM paper_trades")
        db.execute("DELETE FROM paper_decisions")
    result = diagnose(str(path))
    assert result["strategies"] == {}
