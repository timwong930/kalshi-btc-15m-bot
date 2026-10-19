"""Offline regression tests for prospective, quotes-only virtual paper simulator."""
from __future__ import annotations

import json
import os
import sqlite3
import time

import pytest

from src.ml_dataset import FEATURES
from src.paper_sim import (
    Policy,
    _predict_shadow,
    causal_features,
    open_ledger,
    report,
    run_tick,
    save_reports,
    shadow_artifacts,
)
from src.research_stream import ResearchStore

T = (int(time.time() * 1000) // 100000) * 100000
CLOSE = T + 300000
TICKER = "KXBTC15M-TEST"


def prepare(tmp_path):
    research_file = str(tmp_path / "research.db")
    paper_file = str(tmp_path / "paper.db")
    store = ResearchStore(research_file)
    store.market(TICKER, CLOSE, 82000., "greater_or_equal")
    # Simulate an otherwise trade-eligible observation BEFORE startup.
    store.quote(
        TICKER, "kalshi_rest", T + 10000, yes_bid=.25, yes_ask=.30,
        no_bid=.70, no_ask=.75, yes_depth=15, no_depth=15,
    )
    source = sqlite3.connect(research_file)
    source.row_factory = sqlite3.Row
    ledger = open_ledger(paper_file, source)
    return store, source, ledger


def quote(store, t, *, yes_bid=.25, yes_ask=.30):
    store.quote(
        TICKER, "kalshi_rest", t,
        yes_bid=yes_bid, yes_ask=yes_ask, no_bid=1 - yes_ask,
        no_ask=1 - yes_bid, yes_depth=20, no_depth=20,
    )


def test_prospective_entries_exit_math_and_idempotent_ledger(tmp_path):
    store, source, ledger = prepare(tmp_path)
    try:
        assert ledger.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0
        run_tick(source, ledger, str(tmp_path / "none"), Policy(), now_ms=T + 10500)
        assert ledger.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0
        quote(store, T + 10000)
        run_tick(source, ledger, str(tmp_path / "none"), Policy(), now_ms=T + 10500)
        first = ledger.execute(
            "SELECT * FROM paper_trades WHERE strategy='price_rule'"
        ).fetchone()
        assert first["entry_ask"] == pytest.approx(.30)
        assert first["status"] == "OPEN"
        assert len(ledger.execute("SELECT * FROM paper_decisions").fetchall()) == 6
        assert not ledger.execute(
            "SELECT * FROM paper_trades WHERE strategy LIKE 'reversal_%'"
        ).fetchall()
        quote(store, T + 15000, yes_bid=.50, yes_ask=.55)
        run_tick(source, ledger, str(tmp_path / "none"), Policy(), now_ms=T + 15500)
        closed = ledger.execute(
            "SELECT * FROM paper_trades WHERE strategy='price_rule'"
        ).fetchone()
        assert closed["status"] == "CLOSED"
        assert closed["reason"] == "take_profit"
        assert closed["pnl_per_contract"] == pytest.approx(.14)
        before = ledger.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
        run_tick(source, ledger, str(tmp_path / "none"), Policy(), now_ms=T + 16000)
        assert ledger.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == before
        stats = report(ledger)
        assert stats["strategies"]["price_rule"]["wins"] == 1
        assert stats["strategies"]["price_rule"]["net_simulated_pnl_per_one_contract_each"] == .14
        assert stats["no_trade_baseline_pnl"] == 0.0
        assert stats["never_actual_fills"] is True
        save_reports(ledger, str(tmp_path / "report"))
        assert (tmp_path / "report" / "latest_metrics.json").exists()
        assert (tmp_path / "report" / "paper_trades.csv").exists()
        assert (tmp_path / "report" / "equity_curves.csv").exists()
    finally:
        ledger.close()
        source.close()
        store.close()


def test_bad_book_and_stale_quotes_not_used_for_entry(tmp_path):
    store, source, ledger = prepare(tmp_path)
    try:
        quote(store, T + 10000, yes_bid=.25, yes_ask=.30)
        run_tick(source, ledger, "", Policy(), now_ms=T + 50000)
        assert ledger.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0
        # A fresh valid quote can now be used; prior stale quote cannot.
        quote(store, T + 51000)
        run_tick(source, ledger, "", Policy(), now_ms=T + 51500)
        assert ledger.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 1
    finally:
        ledger.close()
        source.close()
        store.close()


def test_gap_is_censored_not_fake_profitable_exit(tmp_path):
    store, source, ledger = prepare(tmp_path)
    try:
        quote(store, T + 10000)
        run_tick(source, ledger, "", Policy(), now_ms=T + 11000)
        quote(store, T + 45001, yes_bid=.75, yes_ask=.80)
        run_tick(source, ledger, "", Policy(), now_ms=T + 45500)
        trade = ledger.execute(
            "SELECT * FROM paper_trades WHERE strategy='price_rule'"
        ).fetchone()
        assert trade["status"] == "CENSORED"
        assert trade["pnl_per_contract"] is None
        stats = report(ledger)["strategies"]["price_rule"]
        assert stats["closed_trades"] == 0
        assert stats["censored_trades"] == 1
    finally:
        ledger.close()
        source.close()
        store.close()


def test_stop_loss_and_no_invalid_expiration_fill(tmp_path):
    store, source, ledger = prepare(tmp_path)
    try:
        quote(store, T + 10000)
        run_tick(source, ledger, "", Policy(), now_ms=T + 10500)
        quote(store, T + 15000, yes_bid=.10, yes_ask=.15)
        run_tick(source, ledger, "", Policy(), now_ms=T + 15500)
        trade = ledger.execute(
            "SELECT * FROM paper_trades WHERE strategy='price_rule'"
        ).fetchone()
        assert trade["status"] == "CLOSED"
        assert trade["reason"] == "stop_loss"
        assert trade["pnl_per_contract"] == pytest.approx(-.26)
    finally:
        ledger.close()
        source.close()
        store.close()

    store, source, ledger = prepare(tmp_path / "next")
    try:
        quote(store, T + 10000)
        run_tick(source, ledger, "", Policy(), now_ms=T + 10500)
        run_tick(source, ledger, "", Policy(), now_ms=CLOSE + 10000)
        trade = ledger.execute(
            "SELECT * FROM paper_trades WHERE strategy='price_rule'"
        ).fetchone()
        assert trade["status"] == "CENSORED"
        assert trade["pnl_per_contract"] is None
    finally:
        ledger.close()
        source.close()
        store.close()


def test_shadows_cannot_use_post_entry_training_or_future_artifact(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    path = models / "outcome_logistic.json"
    data = {
        "algorithm": "offline-logistic-v1",
        "research_task": "outcome",
        "mode": "SHADOW_RESEARCH_ONLY",
        "live_trading_approval": False,
        "trained_through_close_ms": T + 30000,
        "feature_names": list(FEATURES),
        "feature_means": [0.0] * len(FEATURES), "feature_scales": [1.0] * len(FEATURES),
        "weights": [0.0] * len(FEATURES), "intercept": 1.0,
    }
    path.write_text(json.dumps(data))
    os.utime(path, (T / 1000 - 10, T / 1000 - 10))
    assert shadow_artifacts(str(models), T) == {}
    data["trained_through_close_ms"] = T - 900000
    path.write_text(json.dumps(data))
    os.utime(path, (T / 1000 + 10, T / 1000 + 10))
    assert shadow_artifacts(str(models), T) == {}
    os.utime(path, (T / 1000 - 10, T / 1000 - 10))
    artifact, digest = shadow_artifacts(str(models), T)["outcome_logistic"]
    assert len(digest) == 64
    assert _predict_shadow(artifact, (0.2,) * len(FEATURES)) == pytest.approx(.73105858)
    data["live_trading_approval"] = True
    path.write_text(json.dumps(data))
    os.utime(path, (T / 1000 - 10, T / 1000 - 10))
    assert shadow_artifacts(str(models), T) == {}


def test_training_features_do_not_include_future_btc_prices(tmp_path):
    store, source, ledger = prepare(tmp_path)
    try:
        entry = T + 10000
        for i in range(20):
            ts = entry - 65000 + i * 3000
            store.price("coinbase", 82000 + 2 * i, ts, ts, "ticker")
        store.price("coinbase", 93000, entry + 2000, entry + 2000, "ticker")
        quote(store, entry)
        row = source.execute(
            "SELECT * FROM research_quotes ORDER BY id DESC LIMIT 1"
        ).fetchone()
        market = source.execute(
            "SELECT * FROM research_markets WHERE ticker=?", (TICKER,)
        ).fetchone()
        features = causal_features(source, row, market)
        assert features is not None
        assert features[0][1] < 10
    finally:
        ledger.close()
        source.close()
        store.close()


def test_invalid_policy_fails_closed():
    with pytest.raises(ValueError):
        Policy(min_depth=-1)
    with pytest.raises(ValueError):
        Policy(max_quote_gap_ms=10)
