"""Tests for read-only expected-value signals (no orders, network, or credentials)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.edge_signals import evaluate_edge_signal
from src.markets import MarketSnapshot, OrderbookSummary
from src.predictor import Prediction
from tests.conftest import make_settings


def _fixtures(
    *,
    p_yes=0.85,
    yes_bid=0.25,
    no_bid=0.30,
    yes_depth=20,
    no_depth=20,
    volume=100,
    seconds_remaining=300,
    age_seconds=0,
):
    now = datetime.now(timezone.utc)
    when = now - timedelta(seconds=age_seconds)
    s = MarketSnapshot(
        ticker="KXBTC15M-TEST",
        event_ticker="TEST",
        status="active",  # actual API status for an open market
        yes_bid=yes_bid,
        yes_ask=1 - no_bid,
        no_bid=no_bid,
        no_ask=1 - yes_bid,
        last_price=0.50,
        volume=volume,
        volume_24h=volume,
        open_interest=50,
        floor_strike=100000,
        strike_type="greater_or_equal",
        close_time=now + timedelta(seconds=seconds_remaining),
        pulled_at=when,
    )
    prediction = Prediction(
        ticker=s.ticker,
        btc_price=100100,
        floor_strike=100000,
        probability_yes=p_yes,
        confidence=abs(2 * p_yes - 1),
        sample_count=10,
        rationale="synthetic test",
    )
    book = OrderbookSummary(
        ticker=s.ticker,
        yes_levels=[(yes_bid, yes_depth)],
        no_levels=[(no_bid, no_depth)],
        yes_depth_total=yes_depth,
        no_depth_total=no_depth,
    )
    return s, prediction, book, when, now


def _evaluate(**kwargs):
    s, pred, book, ts, now = _fixtures(**kwargs)
    return evaluate_edge_signal(s, pred, book, make_settings(), price_sample_time=ts, now=now)


def test_buy_up_uses_opposite_no_bid_as_executable_yes_ask():
    signal = _evaluate()
    assert signal.action == "BUY UP"
    assert signal.ask_dollars == pytest.approx(0.70)
    assert signal.estimated_net_edge_dollars == pytest.approx(0.13)
    assert signal.available_contracts == 20
    assert "ASSUMED_fee=0.0200" in signal.log_line(0.02)


def test_buy_down_uses_yes_bid_and_no_probability():
    signal = _evaluate(p_yes=0.05, yes_bid=0.70, no_bid=0.10)
    assert signal.action == "BUY DOWN"
    assert signal.ask_dollars == pytest.approx(0.30)
    assert signal.model_probability == pytest.approx(0.95)
    assert signal.estimated_net_edge_dollars == pytest.approx(0.63)


def test_skip_when_edge_below_minimum_after_assumed_fees():
    signal = _evaluate(p_yes=0.73)
    assert signal.action == "SKIP"
    assert "below threshold" in signal.reason


def test_insufficient_top_level_liquidity_skips():
    signal = _evaluate(yes_depth=0, no_depth=2)
    assert signal.action == "SKIP"
    assert "liquid" in signal.reason


def test_stale_market_data_skips():
    signal = _evaluate(age_seconds=40)
    assert signal.action == "SKIP"
    assert "stale" in signal.reason


def test_volume_floor_skips_placeholder_market():
    signal = _evaluate(volume=0)
    assert signal.action == "SKIP"
    assert "volume" in signal.reason


def test_too_close_to_settlement_skips():
    signal = _evaluate(seconds_remaining=10)
    assert signal.action == "SKIP"
    assert "window" in signal.reason


def test_no_prediction_fails_closed():
    s, _pred, book, ts, now = _fixtures()
    signal = evaluate_edge_signal(s, None, book, make_settings(), price_sample_time=ts, now=now)
    assert signal.action == "SKIP"


def test_no_book_or_spot_timestamp_fails_closed():
    s, pred, book, ts, now = _fixtures()
    assert evaluate_edge_signal(s, pred, None, make_settings(), price_sample_time=ts, now=now).action == "SKIP"
    assert evaluate_edge_signal(s, pred, book, make_settings(), price_sample_time=None, now=now).action == "SKIP"


def test_material_snapshot_book_disagreement_skips():
    s, pred, book, ts, now = _fixtures()
    book.no_levels = [(0.45, 20)]  # old snapshot YES ask 70c, book now 55c
    signal = evaluate_edge_signal(s, pred, book, make_settings(), price_sample_time=ts, now=now)
    assert signal.action == "SKIP"


def test_closed_market_never_signals():
    s, pred, book, ts, now = _fixtures()
    s.status = "closed"
    signal = evaluate_edge_signal(s, pred, book, make_settings(), price_sample_time=ts, now=now)
    assert signal.action == "SKIP"


def test_no_credential_or_order_api_calls_needed():
    s, pred, book, ts, now = _fixtures()
    settings = make_settings(env="prod", trading_enabled=False, api_key_id=None, private_key_path=None)
    signal = evaluate_edge_signal(s, pred, book, settings, price_sample_time=ts, now=now)
    assert signal.action == "BUY UP"


def test_configured_fee_and_edge_threshold_apply():
    s, pred, book, ts, now = _fixtures()
    settings = make_settings(signal_fee_per_contract_dollars=0.10, signal_min_net_edge_dollars=0.06)
    signal = evaluate_edge_signal(s, pred, book, settings, price_sample_time=ts, now=now)
    assert signal.action == "SKIP"
