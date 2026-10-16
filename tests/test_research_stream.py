"""Offline parser and conservative paper-replay regression tests (no network)."""
from __future__ import annotations

import base64
import csv
import json
import sqlite3

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from src.research_labels import build_labels, nearest_prior
from src.research_stream import (
    Book,
    ResearchStore,
    backup_research_database,
    parse_brti,
    parse_coinbase_ticker,
    parse_ms,
    read_only_ws_headers,
)


def test_coinbase_ticker_uses_exchange_timestamp_and_ignores_other_products():
    frame = {"type": "ticker", "product_id": "BTC-USD", "price": "82123.45",
             "time": "2026-10-16T06:00:00.100Z"}
    assert parse_coinbase_ticker(frame) == (82123.45, 1792130400100)
    assert parse_coinbase_ticker({**frame, "product_id": "ETH-USD"}) is None
    assert parse_coinbase_ticker({**frame, "price": "nan"}) is None
    assert parse_ms("not a date") is None


def test_cf_benchmarks_parses_only_brti():
    frame = {
        "type": "cfbenchmarks_value",
        "msg": {"index_id": "BRTI",
                "data": json.dumps({"type": "value", "id": "BRTI", "time": 1792130400123, "value": "82121.22"})},
    }
    assert parse_brti(frame) == (82121.22, 1792130400123)
    frame["msg"]["index_id"] = "ETHUSD_RTI"
    assert parse_brti(frame) is None


def test_book_snapshot_delta_and_gap_invalidate_until_new_snapshot():
    book = Book()
    assert book.apply({
        "type": "orderbook_snapshot", "seq": 10,
        "msg": {"yes_dollars_fp": [["0.3500", "12"]], "no_dollars_fp": [["0.6400", "11"]]},
    })
    assert book.quotes()["yes_ask"] == pytest.approx(0.36)
    assert book.quotes()["yes_depth"] == 12
    assert book.apply({
        "type": "orderbook_delta", "seq": 11,
        "msg": {"side": "yes", "price_dollars": "0.3500", "delta_fp": "-3"},
    })
    assert book.quotes()["yes_depth"] == 9
    assert not book.apply({
        "type": "orderbook_delta", "seq": 13,
        "msg": {"side": "yes", "price_dollars": "0.3500", "delta_fp": "1"},
    })
    assert not book.valid
    assert book.quotes()["yes_bid"] is None
    assert book.apply({
        "type": "orderbook_snapshot", "seq": 20,
        "msg": {"yes_dollars_fp": [["0.20", "8"]], "no_dollars_fp": [["0.70", "9"]]},
    })
    assert book.quotes()["yes_ask"] == pytest.approx(0.30)


def _fixture_dbs(tmp_path):
    research_path = str(tmp_path / "research.db")
    outcomes_path = str(tmp_path / "outcomes.db")
    store = ResearchStore(research_path)
    close = 1792131300000
    store.market("KXBTC15M-TEST", close, 82000.0, "greater_or_equal")
    start = close - 600_000
    store.price("coinbase", 82100.0, start - 100, start, "ticker")
    # Both sides can be bought and sold with at least one contract of depth.
    store.quote(
        "KXBTC15M-TEST", "kalshi_rest", start,
        yes_bid=.30, yes_ask=.35, no_bid=.65, no_ask=.70,
        yes_depth=100, no_depth=100,
    )
    for i in range(1, 120):
        price = .55 if i >= 2 else .34
        store.quote(
            "KXBTC15M-TEST", "kalshi_rest", start + i * 5000,
            yes_bid=price, yes_ask=min(0.99, price + .05),
            no_bid=max(.01, 0.95 - price), no_ask=1 - price,
            yes_depth=100, no_depth=100,
        )
    store.close()
    with sqlite3.connect(outcomes_path) as con:
        con.execute("CREATE TABLE settled_outcomes (ticker TEXT, result TEXT)")
        con.execute("INSERT INTO settled_outcomes VALUES (?,?)", ("KXBTC15M-TEST", "yes"))
    return research_path, outcomes_path


def test_no_future_timestamps_leak_into_asof_price():
    assert nearest_prior([(10, 100.0), (20, 200.0)], 15) == 100.0
    assert nearest_prior([(10, 100.0), (20, 200.0)], 9) is None
    assert nearest_prior([(10, 100.0)], 20050) is None


def test_reversal_labels_include_fees_slippage_and_quote_coverage(tmp_path):
    research, outcome = _fixture_dbs(tmp_path)
    export = str(tmp_path / "labels.csv")
    rows, complete = build_labels(
        research, outcome, export, target_profit=.10, stop_loss=.50,
        fee_per_side=.02, slippage_per_side=.01,
        maximum_gap_ms=30000,
    )
    assert rows > 0
    assert complete > 0
    with open(export, newline="") as fh:
        data = list(csv.DictReader(fh))
    first = next(row for row in data if row["side"] == "up")
    assert first["entry_ask"] == "0.35"
    assert first["label"] == "take_profit_first"
    assert first["quote_source"] == "kalshi_rest"
    assert float(first["max_favorable_net"]) >= .10
    assert float(first["hypothetical_hold_to_settlement_net"]) == pytest.approx(.62)


def test_unsettled_markets_do_not_generate_training_labels(tmp_path):
    research, outcome = _fixture_dbs(tmp_path)
    with sqlite3.connect(outcome) as con:
        con.execute("DELETE FROM settled_outcomes")
    path = str(tmp_path / "empty.csv")
    assert build_labels(research, outcome, path) == (0, 0)


def test_incomplete_quote_history_is_not_falsely_called_no_reversal(tmp_path):
    research, outcome = _fixture_dbs(tmp_path)
    with sqlite3.connect(research) as con:
        con.execute(
            "DELETE FROM research_quotes WHERE received_ms > "
            "(SELECT MIN(received_ms) + 30000 FROM research_quotes)"
        )
    rows, complete = build_labels(research, outcome, str(tmp_path / "gaps.csv"))
    assert rows > 0
    assert complete == 0


def test_unsupported_training_parameters_rejected(tmp_path):
    research, outcome = _fixture_dbs(tmp_path)
    with pytest.raises(ValueError):
        build_labels(research, outcome, str(tmp_path / "bad.csv"), fee_per_side=-0.01)


def test_research_store_backup_preserves_rows_without_credentials(tmp_path):
    db = str(tmp_path / "research.db")
    store = ResearchStore(db)
    store.price("coinbase", 82200, 1792130400100, 1792130400110, "ticker")
    store.event("research", "heartbeat")
    store.close()
    backup = backup_research_database(db, str(tmp_path / "backups"))
    assert backup.is_file()
    with sqlite3.connect(str(backup)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM research_prices").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM research_events").fetchone()[0] == 1


@pytest.mark.parametrize("algorithm", ["rsa", "ed25519"])
def test_research_ws_signs_get_handshake_only(tmp_path, algorithm):
    if algorithm == "rsa":
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    else:
        private = ed25519.Ed25519PrivateKey.generate()
    path = tmp_path / "research.pem"
    path.write_bytes(private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    headers = read_only_ws_headers("example-key", str(path))
    payload = (headers["KALSHI-ACCESS-TIMESTAMP"] + "GET" + "/trade-api/ws/v2").encode()
    signature = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    assert headers["KALSHI-ACCESS-KEY"] == "example-key"
    if algorithm == "rsa":
        private.public_key().verify(
            signature, payload,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
    else:
        private.public_key().verify(signature, payload)
