"""Read-only, research-only net-edge signals for the current BTC contract.

The Kalshi orderbook exposes resting BIDS for YES and NO. A YES buyer
crosses the highest NO bid at 1 - NO bid; a NO buyer crosses the highest
YES bid at 1 - YES bid. Displayed last price/midpoint is not an ask.

This module does not authenticate, create orders, or mutate storage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from config.settings import Settings
from src.markets import MarketSnapshot, OrderbookSummary
from src.predictor import Prediction


@dataclass(frozen=True)
class EdgeSignal:
    action: Literal["BUY UP", "BUY DOWN", "SKIP"]
    reason: str
    model_probability: float | None = None
    ask_dollars: float | None = None
    estimated_net_edge_dollars: float | None = None
    available_contracts: float | None = None

    def log_line(self, assumed_fee: float) -> str:
        if self.action == "SKIP":
            return f"SKIP (research only): {self.reason}"
        return (
            f"{self.action} CANDIDATE (research only): "
            f"model={self.model_probability:.1%}, "
            f"executable_ask={self.ask_dollars:.4f}, "
            f"estimated_net_edge={self.estimated_net_edge_dollars:+.4f} dollars/contract, "
            f"top_level_depth={self.available_contracts:g}, "
            f"ASSUMED_fee={assumed_fee:.4f} dollars/contract; "
            "not a validated trading recommendation"
        )


def _ask_from_opposite_bids(levels: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Return executable best ask and top-level depth, NOT midpoint or last trade."""
    valid = [
        (float(price), float(size))
        for price, size in levels
        if math.isfinite(price) and math.isfinite(size) and 0 < price < 1 and size > 0
    ]
    if not valid:
        return None
    best_bid = max(price for price, _ in valid)
    depth = sum(size for price, size in valid if abs(price - best_bid) < 1e-9)
    return 1 - best_bid, depth


def evaluate_edge_signal(
    snapshot: MarketSnapshot,
    prediction: Prediction | None,
    book: OrderbookSummary | None,
    settings: Settings,
    *,
    price_sample_time: datetime | None,
    now: datetime | None = None,
) -> EdgeSignal:
    """Compare model probabilities to executable top-of-book asks, less ASSUMED fees.

    Fails closed on absent, stale, or illiquid data. Positive expected value is
    conditional on model calibration and isn't arbitrage or guaranteed profit.
    """
    now = now or datetime.now(timezone.utc)
    # The status=open QUERY returns market.status="active" in the response.
    if snapshot.status not in ("active", "open"):
        return EdgeSignal("SKIP", "market is not open")
    if prediction is None or prediction.ticker != snapshot.ticker:
        return EdgeSignal("SKIP", "no matching model prediction yet")
    if not math.isfinite(prediction.probability_yes) or not 0 <= prediction.probability_yes <= 1:
        return EdgeSignal("SKIP", "invalid model probability")
    if book is None or book.ticker != snapshot.ticker:
        return EdgeSignal("SKIP", "order book unavailable or not for current market")
    if price_sample_time is None:
        return EdgeSignal("SKIP", "missing BTC feed timestamp")

    max_age = settings.signal_max_data_age_seconds
    for label, ts in (("Kalshi quote", snapshot.pulled_at), ("BTC spot", price_sample_time)):
        age = (now - ts).total_seconds()
        if not math.isfinite(age) or age < -2 or age > max_age:
            return EdgeSignal("SKIP", f"{label} is stale or clock is unsynchronized ({age:.1f}s)")
    seconds_remaining = (snapshot.close_time - now).total_seconds()
    if not settings.trade_window_min_seconds <= seconds_remaining <= settings.trade_window_max_seconds:
        return EdgeSignal("SKIP", "outside eligible trading-time window")
    if snapshot.volume is None or not math.isfinite(snapshot.volume) or snapshot.volume < settings.signal_min_volume:
        return EdgeSignal("SKIP", "insufficient traded market volume")
    fee = settings.signal_fee_per_contract_dollars
    threshold = settings.signal_min_net_edge_dollars
    min_depth = settings.signal_min_top_level_contracts
    if not all(math.isfinite(v) for v in (fee, threshold, min_depth)) or min(fee, threshold, min_depth) < 0:
        return EdgeSignal("SKIP", "invalid signal configuration")

    # YES ask is supplied by NO bids; NO ask is supplied by YES bids.
    up = _ask_from_opposite_bids(book.no_levels)
    down = _ask_from_opposite_bids(book.yes_levels)
    possibilities = []
    for action, quote, model_p, displayed_ask in (
        ("BUY UP", up, prediction.probability_yes, snapshot.yes_ask),
        ("BUY DOWN", down, 1 - prediction.probability_yes, snapshot.no_ask),
    ):
        if quote is None:
            continue
        ask, depth = quote
        if depth < min_depth:
            continue
        # Don't compare a newer order book against a materially different old
        # snapshot/model, even if the snapshots are within the freshness window.
        if displayed_ask is not None and abs(ask - displayed_ask) > 0.03:
            continue
        edge = model_p - ask - fee
        possibilities.append((edge, action, model_p, ask, depth))

    if not possibilities:
        return EdgeSignal("SKIP", "no liquid, consistent executable top-of-book quote")
    edge, action, model_p, ask, depth = max(possibilities, key=lambda item: item[0])
    if edge < threshold:
        return EdgeSignal(
            "SKIP",
            f"best modeled edge {edge:+.4f} dollars/contract below threshold "
            f"{threshold:.4f} (fee assumption {fee:.4f})",
        )
    return EdgeSignal(
        action=action,
        reason="model-minus-ask-minus-assumed-fee exceeds configured threshold",
        model_probability=model_p,
        ask_dollars=ask,
        estimated_net_edge_dollars=edge,
        available_contracts=depth,
    )
