"""Produce offline, fee/slippage-aware ML labels from saved READ-ONLY market quotes.

No network or orders. Never confuses quote-based counterfactuals with observed fills.
One row per eligible 30-second decision point and side; split by MARKET in ML.
"""
from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from config.settings import load_settings


def nearest_prior(
    prices: list[tuple[int, float]],
    timestamp_ms: int,
    max_age_ms: int = 10000,
    timestamps: list[int] | None = None,
):
    if not prices:
        return None
    # Timestamps may duplicate; binary search only the monotonically
    # ordered timestamps, never (time, price) tuples.
    times = timestamps if timestamps is not None else [tick[0] for tick in prices]
    idx = bisect_right(times, timestamp_ms) - 1
    if idx < 0 or timestamp_ms - times[idx] > max_age_ms:
        return None
    return prices[idx][1]


def valid_quote(row: sqlite3.Row, side: str, *, depth: float) -> bool:
    if not row["valid"]:
        return False
    fields = (
        ("yes_ask", "no_depth", "yes_bid", "yes_depth")
        if side == "up" else ("no_ask", "yes_depth", "no_bid", "no_depth")
    )
    ask = row[fields[0]]
    entry_depth = row[fields[1]]
    bid = row[fields[2]]
    exit_depth = row[fields[3]]
    return all(
        v is not None and math.isfinite(v) for v in (ask, entry_depth, bid, exit_depth)
    ) and 0 < ask < 1 and 0 < bid < 1 and entry_depth >= depth and exit_depth >= depth


def _price_rows(conn: sqlite3.Connection, source: str) -> list[tuple[int, float]]:
    return [
        (int(row[0]), float(row[1]))
        for row in conn.execute(
            "SELECT received_ms, price FROM research_prices WHERE source = ? ORDER BY received_ms, id",
            (source,),
        )
    ]


def build_labels(
    research_path: str,
    outcomes_path: str,
    export_path: str,
    *,
    target_profit: float = 0.10,
    stop_loss: float = 0.10,
    fee_per_side: float = 0.02,
    slippage_per_side: float = 0.01,
    minimum_depth: float = 1,
    min_hold_ms: int = 1000,
    sample_interval_ms: int = 30000,
    maximum_gap_ms: int = 30000,
) -> tuple[int, int]:
    """Return (#rows, #complete) and export; all dollar quantities are per contract."""
    if min(target_profit, stop_loss, fee_per_side, slippage_per_side, minimum_depth) < 0:
        raise ValueError("negative fee, profit, loss, slippage or depth")
    if sample_interval_ms < 1000 or maximum_gap_ms < 1000 or min_hold_ms < 0:
        raise ValueError("invalid timing thresholds")
    research = Path(research_path).resolve()
    outcomes = Path(outcomes_path).resolve()
    if not research.is_file() or not outcomes.is_file():
        raise FileNotFoundError("Both the research and existing settled-outcomes databases must exist")
    rdb = sqlite3.connect(f"{research.as_uri()}?mode=ro", uri=True)
    odb = sqlite3.connect(f"{outcomes.as_uri()}?mode=ro", uri=True)
    rdb.row_factory = sqlite3.Row
    odb.row_factory = sqlite3.Row
    try:
        settled = {
            row["ticker"]: row["result"]
            for row in odb.execute("SELECT ticker, result FROM settled_outcomes WHERE result IN ('yes', 'no')")
        }
        prices = {source: _price_rows(rdb, source) for source in ("coinbase", "brti")}
        price_times = {source: [tick[0] for tick in data] for source, data in prices.items()}
        meta = rdb.execute(
            "SELECT ticker, close_ms, strike FROM research_markets ORDER BY close_ms"
        ).fetchall()
        dest = Path(export_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "ticker", "entry_ms", "close_ms", "side", "quote_source", "market_strike",
            "seconds_remaining", "btc_price_asof", "brti_price_asof", "entry_ask",
            "entry_bid", "entry_top_depth", "entry_spread", "assumed_fee_per_side",
            "assumed_slippage_per_side", "target_net_profit", "stop_net_loss",
            "label", "first_exit_ms", "first_exit_bid", "max_favorable_net",
            "max_adverse_net", "complete_quote_coverage", "settled_result",
            "hypothetical_hold_to_settlement_net",
        ]
        count, complete = 0, 0
        with dest.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for market in meta:
                ticker, close_ms, strike = market["ticker"], market["close_ms"], market["strike"]
                # Only settled markets can be safely used as supervised labels.
                if ticker not in settled:
                    continue
                quotes_all = rdb.execute(
                    "SELECT * FROM research_quotes WHERE ticker=? AND received_ms <=? "
                    "AND valid=1 ORDER BY received_ms, id", (ticker, close_ms)
                ).fetchall()
                grouped = defaultdict(list)
                for q in quotes_all:
                    grouped[q["source"]].append(q)
                # Prefer authenticated streaming; never splice two timing regimes into one replay.
                selected = "kalshi_ws" if len(grouped["kalshi_ws"]) >= 10 else "kalshi_rest"
                quotes = grouped[selected]
                if not quotes:
                    continue
                for side in ("up", "down"):
                    sampled_bucket = set()
                    for i, entry in enumerate(quotes):
                        t = entry["received_ms"]
                        remaining = (close_ms - t) / 1000
                        if not 60 <= remaining <= 780 or not valid_quote(entry, side, depth=minimum_depth):
                            continue
                        bucket = t // sample_interval_ms
                        if bucket in sampled_bucket:
                            continue
                        sampled_bucket.add(bucket)
                        if side == "up":
                            ask, bid = entry["yes_ask"], entry["yes_bid"]
                            entry_depth = entry["no_depth"]
                        else:
                            ask, bid = entry["no_ask"], entry["no_bid"]
                            entry_depth = entry["yes_depth"]
                        # Conservative fixed price slippage; depth indicates only displayed interest.
                        total_cost = ask + fee_per_side + slippage_per_side
                        future = [
                            q for q in quotes[i + 1:]
                            if q["received_ms"] >= t + min_hold_ms and valid_quote(q, side, depth=minimum_depth)
                        ]
                        observed = [entry] + future
                        gaps = [
                            b["received_ms"] - a["received_ms"]
                            for a, b in zip(observed, observed[1:])
                        ]
                        covered = bool(future) and all(g <= maximum_gap_ms for g in gaps)
                        covered = covered and close_ms - observed[-1]["received_ms"] <= maximum_gap_ms
                        event, exit_ms, exit_bid = "no_trigger", None, None
                        max_gain, max_loss = None, None
                        for q in future:
                            sell_bid = q["yes_bid"] if side == "up" else q["no_bid"]
                            profit = sell_bid - total_cost - fee_per_side - slippage_per_side
                            max_gain = profit if max_gain is None else max(profit, max_gain)
                            max_loss = profit if max_loss is None else min(profit, max_loss)
                            if event == "no_trigger":
                                if profit >= target_profit:
                                    event, exit_ms, exit_bid = "take_profit_first", q["received_ms"], sell_bid
                                elif profit <= -stop_loss:
                                    event, exit_ms, exit_bid = "stop_loss_first", q["received_ms"], sell_bid
                        if not covered:
                            event = "incomplete_quotes"
                            exit_ms = None
                            exit_bid = None
                        if covered:
                            complete += 1
                        won = settled[ticker] == ("yes" if side == "up" else "no")
                        holding_net = (1.0 if won else 0.0) - ask - fee_per_side - slippage_per_side
                        writer.writerow({
                            "ticker": ticker, "entry_ms": t, "close_ms": close_ms, "side": side,
                            "quote_source": selected, "market_strike": strike,
                            "seconds_remaining": remaining,
                            "btc_price_asof": nearest_prior(prices["coinbase"], t, timestamps=price_times["coinbase"]),
                            "brti_price_asof": nearest_prior(prices["brti"], t, timestamps=price_times["brti"]),
                            "entry_ask": ask, "entry_bid": bid, "entry_top_depth": entry_depth,
                            "entry_spread": ask - bid, "assumed_fee_per_side": fee_per_side,
                            "assumed_slippage_per_side": slippage_per_side,
                            "target_net_profit": target_profit, "stop_net_loss": stop_loss,
                            "label": event, "first_exit_ms": exit_ms, "first_exit_bid": exit_bid,
                            "max_favorable_net": max_gain, "max_adverse_net": max_loss,
                            "complete_quote_coverage": int(covered),
                            "settled_result": settled[ticker],
                            "hypothetical_hold_to_settlement_net": holding_net,
                        })
                        count += 1
        return count, complete
    finally:
        rdb.close()
        odb.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline future reversal labels; NO trades")
    parser.add_argument("--research-db", default="data/research_stream.db")
    parser.add_argument("--outcomes-db", default=None)
    parser.add_argument("--csv", default="data/research/reversal_labels.csv")
    parser.add_argument("--target", type=float, default=0.10)
    parser.add_argument("--stop", type=float, default=0.10)
    parser.add_argument("--fee", type=float, default=0.02, help="Assumed fee per contract per trade")
    parser.add_argument("--slippage", type=float, default=0.01, help="Assumed slippage per side")
    args = parser.parse_args()
    count, complete = build_labels(
        args.research_db, args.outcomes_db or load_settings().db_path, args.csv,
        target_profit=args.target, stop_loss=args.stop, fee_per_side=args.fee,
        slippage_per_side=args.slippage,
    )
    print(f"Exported {count} market-entry rows, {complete} with complete replay coverage to {args.csv}")
    print("No real fills, no orders, no self-training. Split train/test by market and chronology.")


if __name__ == "__main__":
    main()
