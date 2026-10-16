"""Persistence and settlement audits for research-only signals, with no network or orders."""
from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.audit_report import build_summary, export_csv, open_audit_db
from src.backup import export_csv as export_backup_csv
from src.edge_signals import EdgeSignal, evaluate_edge_signal
from src.markets import MarketSnapshot, OrderbookSummary
from src.predictor import Prediction
from src.storage import Storage
from tests.conftest import make_settings


def _inputs(ticker: str = "KXBTC15M-TEST"):
    now = datetime.now(timezone.utc)
    snapshot = MarketSnapshot(
        ticker=ticker,
        event_ticker="TEST",
        status="active",
        yes_bid=0.25,
        yes_ask=0.70,
        no_bid=0.30,
        no_ask=0.75,
        last_price=0.50,
        volume=200,
        volume_24h=1000,
        open_interest=100,
        floor_strike=100000,
        strike_type="greater_or_equal",
        close_time=now + timedelta(minutes=5),
        pulled_at=now,
    )
    prediction = Prediction(
        ticker=ticker, btc_price=100200, floor_strike=100000,
        probability_yes=0.85, confidence=0.70, sample_count=10,
        rationale="test model",
    )
    book = OrderbookSummary(
        ticker=ticker, yes_levels=[(0.25, 20)], no_levels=[(0.30, 20)],
        yes_depth_total=20, no_depth_total=20,
    )
    return now, snapshot, prediction, book


def _record(storage: Storage, *, action: str = "BUY UP", ticker: str = "KXBTC15M-TEST"):
    now, snapshot, prediction, book = _inputs(ticker)
    settings = make_settings()
    decision = evaluate_edge_signal(snapshot, prediction, book, settings, price_sample_time=now, now=now)
    assert decision.action == "BUY UP"
    if action == "SKIP":
        decision = EdgeSignal("SKIP", "manual audit SKIP")
    storage.insert_edge_signal_audit(snapshot, prediction, book, decision, settings, now)
    return snapshot


def test_signal_audit_persists_and_survives_reopening(tmp_path):
    db_path = str(tmp_path / "audit.db")
    storage = Storage(db_path)
    snapshot = _record(storage)
    storage.close()
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT action, signal_version, ticker, snapshot_at_utc, candidate_direction, "
            "candidate_ask, estimated_net_edge, book_no_levels_json, fee_assumed "
            "FROM edge_signal_audit"
        ).fetchone()
    assert row[0] == "BUY UP"
    assert row[1].startswith("edge-v")
    assert row[2] == snapshot.ticker
    assert row[3] == snapshot.pulled_at.isoformat()
    assert row[4] == "BUY UP"
    assert row[5] == pytest.approx(0.70)
    assert row[6] == pytest.approx(0.13)
    assert '"0.3"' not in row[7]  # book saved as numeric list, not string quotes
    assert row[8] == pytest.approx(0.02)
    reopened = Storage(db_path)
    _record(reopened, action="SKIP")
    assert reopened._conn.execute("SELECT COUNT(*) FROM edge_signal_audit").fetchone()[0] == 2
    reopened.close()


def test_settled_outcome_joins_without_mutating_original_decisions(tmp_path):
    db_path = str(tmp_path / "audit.db")
    storage = Storage(db_path)
    snapshot = _record(storage)
    _record(storage)  # two correlated candidates in same market
    _record(storage, action="SKIP")
    storage.upsert_settled_outcome(
        {
            "ticker": snapshot.ticker,
            "close_time": snapshot.close_time.isoformat(),
            "result": "yes",
            "settlement_value_dollars": "1.0000",
        }
    )
    storage.close()
    conn = open_audit_db(db_path)
    s = build_summary(conn)
    assert s["evaluations"] == 3
    assert s["skips"] == 1
    assert s["candidate_evaluations"] == 2
    assert s["candidate_markets"] == 1
    assert s["settled_candidate_markets"] == 1
    assert s["hypothetical_wins"] == 1
    assert s["hypothetical_pnl_per_one_contract_per_market"] == pytest.approx(0.28)
    dest = tmp_path / "exports" / "signals.csv"
    assert export_csv(conn, str(dest)) == 3
    conn.close()
    with dest.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert rows[0]["settled_result"] == "yes"
    assert float(rows[0]["counterfactual_one_contract_pnl_if_filled"]) == pytest.approx(0.28)
    assert rows[-1]["counterfactual_one_contract_pnl_if_filled"] == ""


def test_pending_outcomes_not_counted_and_backup_exports_audit(tmp_path):
    db_path = str(tmp_path / "audit.db")
    storage = Storage(db_path)
    _record(storage)
    storage.close()
    conn = open_audit_db(db_path)
    summary = build_summary(conn)
    assert summary["settled_candidate_markets"] == 0
    conn.close()
    paths = export_backup_csv(db_path, str(tmp_path / "backup"))
    assert any(p.name.startswith("edge_signal_audit_") for p in paths)


def test_missing_prediction_skip_is_preserved(tmp_path):
    storage = Storage(str(tmp_path / "audit.db"))
    now, snap, _pred, book = _inputs()
    signal = evaluate_edge_signal(snap, None, book, make_settings(), price_sample_time=now, now=now)
    assert signal.action == "SKIP"
    storage.insert_edge_signal_audit(snap, None, book, signal, make_settings(), now)
    row = storage._conn.execute(
        "SELECT action, reason, model_probability_yes FROM edge_signal_audit"
    ).fetchone()
    assert row[0] == "SKIP"
    assert "prediction" in row[1]
    assert row[2] is None
    storage.close()
