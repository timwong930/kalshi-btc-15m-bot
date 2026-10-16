"""Append-only research collector. Only GET requests and market-data WebSocket subscriptions.

Keyless default: Coinbase public WS + Kalshi public REST every 5s.
Opt-in Kalshi authenticated WS: dedicated RESEARCH_KALSHI_* credentials.
No orders, portfolio calls, execution APIs, or automatic model changes.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding

from config.settings import load_settings
from src.kalshi_client import KalshiClient
from src.markets import discover_active_market, get_orderbook_summary, get_snapshot

LOG = logging.getLogger("research_stream")
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
KALSHI_WS = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_ms(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000)
    except (ValueError, TypeError):
        return None


class ResearchStore:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS research_prices (
                id INTEGER PRIMARY KEY, received_ms INTEGER NOT NULL, source_ms INTEGER,
                source TEXT NOT NULL, price REAL NOT NULL, latency_ms INTEGER,
                raw_type TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_research_prices ON research_prices(source, received_ms);
            CREATE TABLE IF NOT EXISTS research_markets (
                ticker TEXT PRIMARY KEY, close_ms INTEGER NOT NULL, strike REAL,
                strike_type TEXT, first_seen_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_quotes (
                id INTEGER PRIMARY KEY, received_ms INTEGER NOT NULL, source_ms INTEGER,
                source TEXT NOT NULL, ticker TEXT NOT NULL,
                yes_bid REAL, yes_ask REAL, no_bid REAL, no_ask REAL,
                yes_depth REAL, no_depth REAL, volume REAL,
                sequence INTEGER, valid INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_research_quotes ON research_quotes(ticker, received_ms);
            CREATE TABLE IF NOT EXISTS research_events (
                id INTEGER PRIMARY KEY, received_ms INTEGER NOT NULL, source TEXT NOT NULL,
                ticker TEXT, event_type TEXT NOT NULL, details TEXT
            );
        """)
        self.conn.commit()

    def price(self, source: str, price: float, source_ms: int | None, received_ms: int, kind: str) -> None:
        if not math.isfinite(price) or price <= 0:
            return
        latency = received_ms - source_ms if source_ms is not None else None
        self.conn.execute(
            "INSERT INTO research_prices (received_ms, source_ms, source, price, latency_ms, raw_type) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (received_ms, source_ms, source, price, latency, kind),
        )
        self.conn.commit()

    def market(self, ticker: str, close_ms: int, strike: float | None, strike_type: str | None) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO research_markets VALUES (?, ?, ?, ?, ?)",
            (ticker, close_ms, strike, strike_type, now_ms()),
        )
        self.conn.commit()

    def quote(
        self, ticker: str, source: str, received_ms: int, *, source_ms: int | None = None,
        yes_bid=None, yes_ask=None, no_bid=None, no_ask=None, yes_depth=None,
        no_depth=None, volume=None, sequence=None, valid=True,
    ) -> None:
        self.conn.execute(
            "INSERT INTO research_quotes (received_ms, source_ms, source, ticker, "
            "yes_bid, yes_ask, no_bid, no_ask, yes_depth, no_depth, volume, sequence, valid) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (received_ms, source_ms, source, ticker, yes_bid, yes_ask, no_bid, no_ask,
             yes_depth, no_depth, volume, sequence, int(valid)),
        )
        self.conn.commit()

    def event(self, source: str, kind: str, details: str = "", ticker: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO research_events (received_ms, source, ticker, event_type, details) VALUES (?, ?, ?, ?, ?)",
            (now_ms(), source, ticker, kind, details[:400]),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class Book:
    """Full resting bid book; drops validity on sequence gaps until next snapshot."""

    def __init__(self):
        self.levels = {"yes": {}, "no": {}}
        self.seq = None
        self.valid = False

    def apply(self, frame: dict) -> bool:
        kind = frame.get("type")
        msg = frame.get("msg") or {}
        seq = frame.get("seq")
        if kind == "orderbook_snapshot":
            self.levels = {
                side: {float(p): float(s) for p, s in msg.get(f"{side}_dollars_fp", []) if float(s) > 0}
                for side in ("yes", "no")
            }
            self.seq = seq
            self.valid = True
            return True
        if kind != "orderbook_delta":
            return False
        if not self.valid or self.seq is None or not isinstance(seq, int) or seq != self.seq + 1:
            self.valid = False
            return False
        side = msg.get("side")
        if side not in self.levels:
            self.valid = False
            return False
        price = float(msg["price_dollars"])
        amount = self.levels[side].get(price, 0) + float(msg["delta_fp"])
        if amount > 0:
            self.levels[side][price] = amount
        else:
            self.levels[side].pop(price, None)
        self.seq = seq
        return True

    def best(self, side: str):
        book = self.levels[side]
        if not book or not self.valid:
            return None, None
        price = max(book)
        return price, book[price]

    def quotes(self) -> dict:
        yes_bid, yes_depth = self.best("yes")
        no_bid, no_depth = self.best("no")
        return dict(
            yes_bid=yes_bid, no_bid=no_bid,
            yes_ask=1 - no_bid if no_bid is not None else None,
            no_ask=1 - yes_bid if yes_bid is not None else None,
            yes_depth=yes_depth, no_depth=no_depth,
        )


def parse_coinbase_ticker(frame: dict) -> tuple[float, int | None] | None:
    if frame.get("type") != "ticker" or frame.get("product_id") != "BTC-USD":
        return None
    try:
        price = float(frame["price"])
        if not math.isfinite(price) or price <= 0:
            return None
        return price, parse_ms(frame.get("time"))
    except (KeyError, TypeError, ValueError):
        return None


def parse_brti(frame: dict) -> tuple[float, int | None] | None:
    if frame.get("type") != "cfbenchmarks_value":
        return None
    msg = frame.get("msg") or {}
    if msg.get("index_id") != "BRTI":
        return None
    try:
        upstream = json.loads(msg["data"])
        price = float(upstream["value"])
        return price, parse_ms(upstream.get("time"))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def read_only_ws_headers(key_id: str, pem_path: str) -> dict[str, str]:
    """Sign only GET /trade-api/ws/v2, supporting RSA-PSS or Ed25519."""
    with open(pem_path, "rb") as fh:
        private = serialization.load_pem_private_key(fh.read(), password=None)
    timestamp = str(now_ms())
    payload = (timestamp + "GET" + "/trade-api/ws/v2").encode()
    if isinstance(private, ed25519.Ed25519PrivateKey):
        signature = private.sign(payload)
    else:
        signature = private.sign(
            payload,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
    }


async def coinbase_loop(store: ResearchStore) -> None:
    retry_seconds = 1
    while True:
        try:
            async with websockets.connect(COINBASE_WS, ping_interval=20, open_timeout=15) as sock:
                await sock.send(json.dumps({"type": "subscribe", "product_ids": ["BTC-USD"],
                                            "channels": ["ticker", "heartbeat"]}))
                store.event("coinbase", "connected")
                retry_seconds = 1
                async for text_frame in sock:
                    received = now_ms()
                    frame = json.loads(text_frame)
                    tick = parse_coinbase_ticker(frame)
                    if tick:
                        store.price("coinbase", tick[0], tick[1], received, "ticker")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.warning("Coinbase stream reconnecting: %s", type(exc).__name__)
            store.event("coinbase", "disconnected", type(exc).__name__)
        await asyncio.sleep(retry_seconds)
        retry_seconds = min(30, retry_seconds * 2)


async def kalshi_rest_loop(store: ResearchStore, poll_seconds: float, current_market: dict) -> None:
    # Explicitly remove credentials even if the user's .env has production keys.
    settings = replace(load_settings(), env="prod",
                       base_url="https://external-api.kalshi.com/trade-api/v2",
                       api_key_id=None, private_key_path=None)
    client = KalshiClient(settings)
    while True:
        try:
            market = await asyncio.to_thread(discover_active_market, client, "KXBTC15M")
            if market:
                ticker = market["ticker"]
                snapshot = await asyncio.to_thread(get_snapshot, client, ticker)
                book = await asyncio.to_thread(get_orderbook_summary, client, ticker)
                current_market["ticker"] = ticker
                store.market(ticker, int(snapshot.close_time.timestamp() * 1000),
                             snapshot.floor_strike, snapshot.strike_type)
                top_yes = max(book.yes_levels, default=(None, None))
                top_no = max(book.no_levels, default=(None, None))
                # Quote only from the fetched bid book: a later/earlier market
                # snapshot must not be misrepresented as a tradable ask.
                yes_bid, yes_depth = top_yes
                no_bid, no_depth = top_no
                store.quote(
                    ticker, "kalshi_rest", now_ms(),
                    yes_bid=yes_bid, no_bid=no_bid,
                    yes_ask=(1 - no_bid) if no_bid is not None else None,
                    no_ask=(1 - yes_bid) if yes_bid is not None else None,
                    yes_depth=yes_depth, no_depth=no_depth, volume=snapshot.volume,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.warning("Kalshi REST poll failed: %s", type(exc).__name__)
            store.event("kalshi_rest", "poll_failed", type(exc).__name__)
        await asyncio.sleep(poll_seconds)


async def kalshi_ws_loop(store: ResearchStore, current_market: dict, key_id: str, pem_path: str) -> None:
    """Optional authenticated market-data ONLY (never uses KalshiClient.post)."""
    retry_seconds = 1
    while True:
        ticker = current_market.get("ticker")
        if not ticker:
            await asyncio.sleep(2)
            continue
        try:
            headers = read_only_ws_headers(key_id, pem_path)
            async with websockets.connect(KALSHI_WS, additional_headers=headers,
                                          ping_interval=20, open_timeout=15) as sock:
                book = Book()
                await sock.send(json.dumps({
                    "id": 1, "cmd": "subscribe",
                    "params": {"channels": ["orderbook_delta", "trade"], "market_tickers": [ticker]},
                }))
                await sock.send(json.dumps({
                    "id": 2, "cmd": "subscribe",
                    "params": {"channels": ["cfbenchmarks_value"], "index_ids": ["BRTI"]},
                }))
                store.event("kalshi_ws", "connected", ticker=ticker)
                retry_seconds = 1
                while current_market.get("ticker") == ticker:
                    try:
                        raw = await asyncio.wait_for(sock.recv(), timeout=3)
                    except asyncio.TimeoutError:
                        continue
                    received = now_ms()
                    frame = json.loads(raw)
                    kind = frame.get("type")
                    if kind == "error":
                        store.event("kalshi_ws", "subscription_error",
                                    str((frame.get("msg") or {}).get("msg", "unknown")), ticker)
                    brti = parse_brti(frame)
                    if brti:
                        store.price("brti", brti[0], brti[1], received, kind)
                    if kind in ("orderbook_snapshot", "orderbook_delta"):
                        if (frame.get("msg") or {}).get("market_ticker") != ticker:
                            continue
                        if not book.apply(frame):
                            store.event("kalshi_ws", "sequence_gap_or_unsynced", ticker=ticker)
                            # Reconnect for a fresh snapshot; never label a corrupt book valid.
                            break
                        store.quote(ticker, "kalshi_ws", received,
                                    source_ms=frame.get("sending_ts_ms"),
                                    sequence=frame.get("seq"), **book.quotes())
                    elif kind == "trade" and (frame.get("msg") or {}).get("market_ticker") == ticker:
                        store.event("kalshi_ws", "public_trade", json.dumps(frame.get("msg", {}))[:400], ticker)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.warning("Kalshi WS reconnecting: %s", type(exc).__name__)
            store.event("kalshi_ws", "disconnected", type(exc).__name__, ticker)
        await asyncio.sleep(retry_seconds)
        retry_seconds = min(30, retry_seconds * 2)


async def run_collector(database: str, poll_seconds: float, kalshi_ws: bool, key_id: str, pem_path: str) -> None:
    store = ResearchStore(database)
    active: dict = {}
    LOG.info("READ ONLY: coinbase WS + Kalshi REST%s; data=%s",
             " + opt-in authenticated Kalshi WS/BRTI" if kalshi_ws else "", database)
    tasks = [
        asyncio.create_task(coinbase_loop(store)),
        asyncio.create_task(kalshi_rest_loop(store, poll_seconds, active)),
    ]
    if kalshi_ws:
        tasks.append(asyncio.create_task(kalshi_ws_loop(store, active, key_id, pem_path)))
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        store.close()


def print_status(path: str) -> None:
    db = Path(path).resolve()
    if not db.is_file():
        print(f"No research database yet: {db}")
        return
    conn = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True)
    try:
        for table in ("research_prices", "research_quotes", "research_markets", "research_events"):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table}: {count:,}")
        for source, count, last_ms in conn.execute(
            "SELECT source, COUNT(*), MAX(received_ms) FROM research_prices GROUP BY source"
        ):
            print(f"price source {source}: {count:,} ticks; last receipt ms={last_ms}")
        for source, count, last_ms in conn.execute(
            "SELECT source, COUNT(*), MAX(received_ms) FROM research_quotes GROUP BY source"
        ):
            print(f"quote source {source}: {count:,} updates; last receipt ms={last_ms}")
        for timestamp, source, event_type, details in conn.execute(
            "SELECT received_ms, source, event_type, details FROM research_events "
            "ORDER BY id DESC LIMIT 5"
        ):
            print(f"event {timestamp} {source} {event_type}: {details}")
    finally:
        conn.close()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Read-only BTC research collector; NEVER places orders")
    parser.add_argument("--db", default=os.getenv("RESEARCH_DB_PATH", "data/research_stream.db"))
    parser.add_argument("--rest-seconds", type=float, default=float(os.getenv("RESEARCH_REST_SECONDS", "5")))
    parser.add_argument("--kalshi-ws", action="store_true", help="Opt into signed Kalshi market-data WS")
    parser.add_argument("--status", action="store_true", help="Inspect current research DB and exit")
    args = parser.parse_args()
    if args.status:
        print_status(args.db)
        return
    if args.rest_seconds < 5:
        parser.error("--rest-seconds must be >=5 to avoid excessive REST polling")
    key_id = os.getenv("RESEARCH_KALSHI_API_KEY_ID", "")
    pem_path = os.getenv("RESEARCH_KALSHI_PRIVATE_KEY_PATH", "")
    if args.kalshi_ws and (not key_id or not pem_path):
        parser.error("--kalshi-ws requires dedicated RESEARCH_KALSHI_API_KEY_ID and PRIVATE_KEY_PATH")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(run_collector(args.db, args.rest_seconds, args.kalshi_ws, key_id, pem_path))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
