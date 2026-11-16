"""Phase 2: label and validate historical research snapshots for ML.

Only reads the collector and settlement databases; no network, order, or
model-serving access. One pre-declared observation near T-5m per market avoids
mistaking repeated ticks for independent market outcomes.
"""
from __future__ import annotations

import csv
import math
import sqlite3
import time
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def project_path(path: str | Path) -> Path:
    """Anchor relative CLI paths to this project, avoiding cwd/TCC failures."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def read_only(path: str) -> sqlite3.Connection:
    target = project_path(path)
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


def stream_quality(
    research_db: str,
    *,
    now_ms: int | None = None,
    max_quote_age_ms: int = 120_000,
    max_source_latency_ms: int = 10_000,
    require_kalshi_ws: bool = False,
) -> dict:
    """Summarize source freshness and fail closed on stale quote evidence.

    REST and WebSocket are equivalent market-data evidence for this gate;
    pass ``require_kalshi_ws=True`` only for experiments that specifically
    require WebSocket-only evidence.
    """
    observed_ms = int(time.time() * 1000) if now_ms is None else now_ms
    expected_sources = ("kalshi_ws", "kalshi_rest", "coinbase", "brti")
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
        quote_quality = {
            row["source"]: row
            for row in db.execute(
                """
                SELECT source, COUNT(*) AS count, MAX(received_ms) AS latest_received_ms,
                       MAX(CASE WHEN source_ms IS NOT NULL
                                THEN received_ms - source_ms END) AS max_latency_ms,
                       SUM(CASE WHEN source_ms IS NULL THEN 1 ELSE 0 END) AS missing_source_ms,
                       (SELECT source_ms FROM research_quotes newest
                        WHERE newest.source = quotes.source
                        ORDER BY newest.received_ms DESC, newest.id DESC LIMIT 1)
                        AS latest_source_ms
                FROM research_quotes quotes GROUP BY source
                """
            )
        }
        price_quality = {
            row["source"]: row
            for row in db.execute(
                """
                SELECT source, COUNT(*) AS count, MAX(received_ms) AS latest_received_ms,
                       MAX(latency_ms) AS max_latency_ms,
                       SUM(CASE WHEN source_ms IS NULL THEN 1 ELSE 0 END) AS missing_source_ms
                FROM research_prices GROUP BY source
                """
            )
        }
        source_quality = {}
        for source in expected_sources:
            quote = quote_quality.get(source)
            price = price_quality.get(source)
            latest_quote = quote["latest_received_ms"] if quote else None
            latest_price = price["latest_received_ms"] if price else None
            latest = max(x for x in (latest_quote, latest_price) if x is not None) \
                if any(x is not None for x in (latest_quote, latest_price)) else None
            latest_quote_latency = (
                latest_quote - quote["latest_source_ms"]
                if quote and latest_quote is not None and quote["latest_source_ms"] is not None
                else None
            )
            source_quality[source] = {
                "quote_count": int(quote["count"]) if quote else 0,
                "price_count": int(price["count"]) if price else 0,
                "latest_quote_received_ms": latest_quote,
                "latest_price_received_ms": latest_price,
                "latest_received_ms": latest,
                "quote_age_ms": observed_ms - latest_quote if latest_quote is not None else None,
                "price_age_ms": observed_ms - latest_price if latest_price is not None else None,
                "age_ms": observed_ms - latest if latest is not None else None,
                "latest_quote_latency_ms": latest_quote_latency,
                "max_latency_ms": max(
                    [x["max_latency_ms"] for x in (quote, price) if x and x["max_latency_ms"] is not None],
                    default=None,
                ),
                "missing_source_ms": sum(
                    int(x["missing_source_ms"] or 0) for x in (quote, price) if x
                ),
                "latest_quote_source_ms_missing": bool(
                    quote and quote["latest_source_ms"] is None
                ),
            }
        candidates = ("kalshi_ws",) if require_kalshi_ws else ("kalshi_ws", "kalshi_rest")
        eligible = []
        for source in candidates:
            quality = source_quality[source]
            if (
                quality["quote_count"] > 0
                and quality["quote_age_ms"] is not None
                and 0 <= quality["quote_age_ms"] <= max_quote_age_ms
                and (
                    source == "kalshi_rest"
                    or (
                        not quality["latest_quote_source_ms_missing"]
                        and quality["latest_quote_latency_ms"] is not None
                        and 0 <= quality["latest_quote_latency_ms"] <= max_source_latency_ms
                    )
                )
            ):
                eligible.append(source)
        reasons = []
        if not eligible:
            label = "kalshi_ws" if require_kalshi_ws else "kalshi_rest_or_ws"
            if not any(source_quality[s]["quote_count"] for s in candidates):
                reasons.append(f"missing_required_quote_source:{label}")
            else:
                for source in candidates:
                    quality = source_quality[source]
                    if quality["quote_count"] and source not in eligible:
                        reasons.append(
                            "kalshi_ws_quote_stale" if source == "kalshi_ws"
                            else "kalshi_rest_quote_stale"
                        )
                        if quality["latest_quote_source_ms_missing"]:
                            reasons.append(
                                "kalshi_ws_source_latency_missing" if source == "kalshi_ws"
                                else "kalshi_rest_source_latency_missing"
                            )
                        if (
                            quality["latest_quote_latency_ms"] is not None
                            and quality["latest_quote_latency_ms"] > max_source_latency_ms
                        ):
                            reasons.append(
                                "kalshi_ws_source_latency_exceeded" if source == "kalshi_ws"
                                else "kalshi_rest_source_latency_exceeded"
                            )
        sources["source_quality"] = source_quality
        sources["freshness_gate"] = {
            "ready": not reasons,
            "required_quote_source": "kalshi_ws" if require_kalshi_ws else "kalshi_rest_or_ws",
            "selected_quote_source": eligible[0] if eligible else None,
            "max_quote_age_ms": max_quote_age_ms,
            "max_source_latency_ms": max_source_latency_ms,
            "observed_ms": observed_ms,
            "reasons": reasons,
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
