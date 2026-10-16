"""Phase 2: label and validate historical research snapshots for ML.

Only reads the collector and settlement databases; no network, order, or
model-serving access. One pre-declared observation near T-5m per market avoids
mistaking repeated ticks for independent market outcomes.
"""
from __future__ import annotations

import csv
import math
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev

FEATURES = (
    "market_yes_mid", "spot_vs_strike_bps", "time_left_fraction",
    "yes_spread", "log_yes_depth", "log_no_depth", "spot_volatility_60s",
    "using_brti",
)
REVERSAL_FEATURES = (
    "entry_ask", "entry_spread", "entry_depth_log", "time_left_fraction",
    "spot_vs_strike_bps", "side_is_up",
)


@dataclass(frozen=True)
class Example:
    ticker: str
    close_ms: int
    entry_ms: int
    y: int
    features: tuple[float, ...]
    baseline: float
    yes_ask: float | None = None
    no_ask: float | None = None
    side: str = ""
    label: str = ""


def read_only(path: str) -> sqlite3.Connection:
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Database does not exist: {target}")
    conn = sqlite3.connect(f"{target.as_uri()}?mode=ro", uri=True, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def _asof(conn: sqlite3.Connection, source: str, timestamp: int, max_age_ms: int = 10000):
    row = conn.execute(
        "SELECT received_ms, price FROM research_prices WHERE source=? AND received_ms<=? "
        "ORDER BY received_ms DESC, id DESC LIMIT 1", (source, timestamp),
    ).fetchone()
    if not row or timestamp - row["received_ms"] > max_age_ms:
        return None
    price = row["price"]
    return price if price is not None and math.isfinite(price) and price > 0 else None


def _vol_60s(conn: sqlite3.Connection, timestamp: int) -> float | None:
    # This query includes *only past* local receive timestamps, never future data.
    rows = conn.execute(
        "SELECT received_ms, price FROM research_prices "
        "WHERE source='coinbase' AND received_ms BETWEEN ? AND ? "
        "ORDER BY received_ms DESC, id DESC LIMIT 500",
        (timestamp - 60000, timestamp),
    ).fetchall()
    rows = list(reversed(rows))
    ret = [
        math.log(b["price"] / a["price"])
        for a, b in zip(rows, rows[1:])
        if a["price"] and b["price"] and a["price"] > 0 and b["price"] > 0
    ]
    return pstdev(ret) if len(ret) >= 5 else None


def _clean_quote(quote: sqlite3.Row) -> bool:
    try:
        required = (
            quote["yes_bid"], quote["yes_ask"], quote["no_bid"], quote["no_ask"],
            quote["yes_depth"], quote["no_depth"],
        )
        if not quote["valid"] or any(x is None or not math.isfinite(x) for x in required):
            return False
        yes_bid, yes_ask, no_bid, no_ask, yes_depth, no_depth = required
        return (0 < yes_bid <= yes_ask < 1 and 0 < no_bid <= no_ask < 1
                and min(yes_depth, no_depth) >= 1
                and yes_ask - yes_bid <= 0.30)
    except (ValueError, TypeError):
        return False


def collect_outcome_examples(
    research_db: str, outcomes_db: str,
) -> tuple[list[Example], dict]:
    """One observation near 5m before close per settled market; verify all inputs."""
    with closing(read_only(research_db)) as research, closing(read_only(outcomes_db)) as outcomes:
        settled = {
            row["ticker"]: row["result"]
            for row in outcomes.execute(
                "SELECT ticker, result FROM settled_outcomes WHERE result IN ('yes', 'no')"
            )
        }
        markets = research.execute(
            "SELECT ticker, close_ms, strike, strike_type FROM research_markets ORDER BY close_ms, ticker"
        ).fetchall()
        reasons = Counter()
        examples = []
        for market in markets:
            ticker, close_ms, strike = market["ticker"], market["close_ms"], market["strike"]
            if ticker not in settled:
                reasons["unsettled"] += 1
                continue
            if market["strike_type"] != "greater_or_equal":
                reasons["unsupported_strike_type"] += 1
                continue
            if strike is None or not math.isfinite(strike) or strike <= 0:
                reasons["invalid_strike"] += 1
                continue
            # Fixed pre-declared T-5m observation range; never picks a
            # post-settlement quote and never selects based on the outcome.
            quotes = research.execute(
                "SELECT * FROM research_quotes WHERE ticker=? AND received_ms BETWEEN ? AND ? "
                "ORDER BY ABS(received_ms - ?), received_ms, id LIMIT 30",
                (ticker, close_ms - 420000, close_ms - 240000, close_ms - 300000),
            ).fetchall()
            quote = next((q for q in quotes if _clean_quote(q)), None)
            if quote is None:
                reasons["no_usable_t_minus_5m_quote"] += 1
                continue
            t = quote["received_ms"]
            coinbase = _asof(research, "coinbase", t)
            brti = _asof(research, "brti", t)
            spot = brti if brti is not None else coinbase
            if spot is None:
                reasons["stale_or_missing_btc_tick"] += 1
                continue
            vol = _vol_60s(research, t)
            if vol is None:
                reasons["missing_60s_volatility"] += 1
                continue
            mid = (quote["yes_bid"] + quote["yes_ask"]) / 2
            p = min(0.999, max(0.001, mid))
            features = (
                p,
                max(-200.0, min(200.0, (spot - strike) / spot * 10000)),
                (close_ms - t) / 900000.0,
                quote["yes_ask"] - quote["yes_bid"],
                math.log1p(quote["yes_depth"]),
                math.log1p(quote["no_depth"]),
                vol,
                float(brti is not None),
            )
            if not all(math.isfinite(x) for x in features):
                reasons["invalid_feature"] += 1
                continue
            examples.append(Example(
                ticker=ticker, close_ms=close_ms, entry_ms=t,
                y=int(settled[ticker] == "yes"), features=features, baseline=p,
                yes_ask=quote["yes_ask"], no_ask=quote["no_ask"],
            ))
        return examples, {
            "total_research_markets": len(markets),
            "settled_outcomes_available": len(settled),
            "usable_settled_markets": len(examples),
            "excluded_market_reasons": dict(sorted(reasons.items())),
            "observation_policy": "single quote near T-5min; past data only; depth>=1",
        }


def load_reversal_examples(csv_path: str, max_seconds_from_target: float = 100) -> tuple[list[Example], dict]:
    """Use ONLY replay rows with complete quote coverage and valid prior labels.

    Data grouped by ticker for splitting, even though two opposing sides
    may share the same settlement and timestamp.
    """
    path = Path(csv_path)
    if not path.exists():
        return [], {"complete_reversal_markets": 0, "reversal_rows": 0}
    chosen = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["complete_quote_coverage"] != "1":
                continue
            if row["label"] not in ("take_profit_first", "stop_loss_first", "no_trigger"):
                continue
            try:
                seconds = float(row["seconds_remaining"])
                if abs(seconds - 300) > max_seconds_from_target:
                    continue
                ticker, side = row["ticker"], row["side"]
                if side not in ("up", "down"):
                    continue
                key = (ticker, side)
                if key not in chosen or abs(seconds - 300) < abs(float(chosen[key]["seconds_remaining"]) - 300):
                    chosen[key] = row
            except (KeyError, ValueError):
                continue
    examples = []
    for (ticker, side), row in chosen.items():
        try:
            ask = float(row["entry_ask"])
            spread = float(row["entry_spread"])
            depth = float(row["entry_top_depth"])
            strike = float(row["market_strike"])
            spot_text = row["brti_price_asof"] or row["btc_price_asof"]
            spot = float(spot_text)
            entry_ms, close_ms = int(row["entry_ms"]), int(row["close_ms"])
            if not (0 < ask < 1 and depth >= 1 and spot > 0 and math.isfinite(strike)):
                continue
            features = (
                ask, spread, math.log1p(depth),
                (close_ms - entry_ms) / 900000.0,
                max(-200.0, min(200.0, (spot - strike) / spot * 10000))
                * (1 if side == "up" else -1),
                float(side == "up"),
            )
            if all(math.isfinite(v) for v in features):
                examples.append(Example(
                    ticker=ticker, close_ms=close_ms, entry_ms=entry_ms,
                    y=int(row["label"] == "take_profit_first"),
                    features=features, baseline=0.5,
                    side=side, label=row["label"],
                ))
        except (KeyError, ValueError, TypeError, OverflowError):
            continue
    return sorted(examples, key=lambda x: (x.close_ms, x.ticker, x.side)), {
        "complete_reversal_markets": len({x.ticker for x in examples}),
        "reversal_rows": len(examples),
        "reversal_outcome": "take-profit reached before stop-loss or timeout",
    }


def stream_quality(research_db: str) -> dict:
    with closing(read_only(research_db)) as db:
        sources = {
            "price_" + r["source"]: r["n"]
            for r in db.execute(
                "SELECT source, count(*) AS n FROM research_prices GROUP BY source"
            )
        }
        sources.update({
            "quote_" + r["source"]: r["n"]
            for r in db.execute(
                "SELECT source, count(*) AS n FROM research_quotes GROUP BY source"
            )
        })
        # Gaps are measured per-market/per-source, so a normal 15-minute
        # market rollover is not counted as a network outage.
        gap_rows = db.execute(
            """
            WITH history AS (
              SELECT source, ticker, received_ms,
                     LAG(received_ms) OVER (
                       PARTITION BY source, ticker ORDER BY received_ms, id
                     ) AS previous_ms
              FROM research_quotes WHERE valid=1
            )
            SELECT source, MAX(received_ms - previous_ms) AS max_gap_ms,
                   SUM(CASE WHEN received_ms - previous_ms > 30000 THEN 1 ELSE 0 END) AS gaps_gt_30s
            FROM history GROUP BY source
            """
        ).fetchall()
        sources["per_market_quote_gaps"] = {
            row["source"]: {
                "max_gap_ms": row["max_gap_ms"],
                "gaps_gt_30s": row["gaps_gt_30s"] or 0,
            }
            for row in gap_rows
        }
        return sources


def write_examples(examples: list[Example], output: str, feature_names: tuple[str, ...]) -> None:
    """Write only historical (settled) records; no secrets or future input features."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ticker", "close_ms", "entry_ms", "target", "baseline", *feature_names])
        for x in examples:
            writer.writerow([x.ticker, x.close_ms, x.entry_ms, x.y, x.baseline, *x.features])
