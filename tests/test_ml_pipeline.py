"""Phase 2+3 local-only regression coverage; no exchange requests or order calls."""
from __future__ import annotations

import csv
import json
import sqlite3

import pytest

from src.ml_dataset import (
    FEATURES,
    Example,
    collect_outcome_examples,
    load_reversal_examples,
)
from src.ml_models import final_shadow_model, fit_model, grouped_markets, walk_forward
from src.ml_pipeline import run_cycle
from src.research_stream import ResearchStore

START_MS = 1792131300000
STRIKE = 82000.0


def _dbs(tmp_path, *, missing_spot=False, unsettled=False):
    research_path = str(tmp_path / "research.db")
    outcomes_path = str(tmp_path / "original.db")
    collector = ResearchStore(research_path)
    close = START_MS + 900000
    ticker = "KXBTC15M-TEST"
    collector.market(ticker, close, STRIKE, "greater_or_equal")
    entry = close - 300000
    # Historical ticks strictly before the decision, enough to compute 60s volatility.
    if not missing_spot:
        for idx in range(16):
            timestamp = entry - 59000 + idx * 3500
            collector.price("coinbase", STRIKE + idx * 2, timestamp, timestamp, "ticker")
    collector.quote(
        ticker, "kalshi_rest", entry,
        yes_bid=0.42, yes_ask=0.47, no_bid=0.53, no_ask=0.58,
        yes_depth=20, no_depth=18, valid=True,
    )
    # FUTURE quote / future BTC price must not leak into input features.
    collector.price("coinbase", STRIKE + 10000, entry + 1000, entry + 1000, "ticker")
    collector.quote(
        ticker, "kalshi_rest", entry + 1000,
        yes_bid=0.99, yes_ask=0.99, no_bid=0.01, no_ask=0.01,
        yes_depth=50, no_depth=50, valid=True,
    )
    collector.close()
    with sqlite3.connect(outcomes_path) as con:
        con.execute("CREATE TABLE settled_outcomes (ticker TEXT PRIMARY KEY, result TEXT)")
        if not unsettled:
            con.execute("INSERT INTO settled_outcomes VALUES (?, ?)", (ticker, "yes"))
    return research_path, outcomes_path, ticker


def test_historical_features_use_only_prior_timestamps(tmp_path):
    research, outcomes, ticker = _dbs(tmp_path)
    examples, quality = collect_outcome_examples(research, outcomes)
    assert quality["usable_settled_markets"] == 1
    assert len(examples) == 1
    example = examples[0]
    assert example.ticker == ticker
    assert example.y == 1
    assert example.features[0] == pytest.approx(.445)
    # The +$10,000 BTC tick one second AFTER the decision must not be selected.
    assert 0 < example.features[1] < 10
    assert len(example.features) == len(FEATURES)
    assert example.entry_ms == example.close_ms - 300000


def test_missing_or_unsettled_observations_are_excluded(tmp_path):
    research, outcomes, _ = _dbs(tmp_path, missing_spot=True)
    examples, quality = collect_outcome_examples(research, outcomes)
    assert examples == []
    assert quality["excluded_market_reasons"]["stale_or_missing_btc_tick"] == 1
    with sqlite3.connect(outcomes) as con:
        con.execute("DELETE FROM settled_outcomes")
    examples, quality = collect_outcome_examples(research, outcomes)
    assert examples == []
    assert quality["excluded_market_reasons"]["unsettled"] == 1


def test_market_grouped_walk_forward_keeps_test_market_out_of_training():
    examples = []
    for i in range(104):
        close = START_MS + 900000 * (i + 1)
        for side in ("up", "down"):
            examples.append(Example(
                ticker=f"TICKER-{i}", close_ms=close, entry_ms=close - 300000,
                y=int(i % 3 == 0) if side == "up" else int(i % 3 != 0),
                features=(.1 + .7 * (i % 2), i / 104, float(side == "up")),
                baseline=.5, side=side,
            ))
    assert len(grouped_markets(examples)) == 104
    for algorithm in ("logistic", "boosted_stumps"):
        result = walk_forward(
            examples, min_train_markets=60, fold_test_markets=20,
            baseline_type="prevalence", algorithm=algorithm,
        )
        assert result["status"] == "SHADOW_EVALUATED"
        assert result["tested_markets"] == 40
        assert result["folds"] == 2
        for fold in result["fold_details"]:
            assert fold["train_last_close_ms"] < fold["test_first_entry_ms"]
        assert result["approval"] == "RESEARCH_ONLY_NEVER_AUTO_PROMOTE"
        model = fit_model(examples, algorithm)
        assert 0 < model.predict(examples[0].features) < 1
        assert final_shadow_model(examples, ("a", "b", "c"), algorithm)["live_trading_approval"] is False


def test_insufficient_markets_fails_closed():
    examples = [
        Example(ticker=f"t{i}", close_ms=START_MS + i * 900000, entry_ms=START_MS + i * 900000 - 300000,
                y=i % 2, baseline=.5, features=(.1, i / 12))
        for i in range(15)
    ]
    for algorithm in ("logistic", "boosted_stumps"):
        metrics = walk_forward(examples, algorithm=algorithm)
        assert metrics["status"] == "INSUFFICIENT_DATA"
        assert metrics["tested_markets"] == 0
        assert metrics["markets_needed_for_first_fold"] == 65


def test_incomplete_reversal_history_cannot_train(tmp_path):
    path = tmp_path / "reversal.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "ticker", "side", "complete_quote_coverage", "label", "seconds_remaining",
            "entry_ask", "entry_spread", "entry_top_depth", "market_strike",
            "brti_price_asof", "btc_price_asof", "entry_ms", "close_ms",
        ])
        writer.writeheader()
        writer.writerow({
            "ticker": "abc", "side": "up", "complete_quote_coverage": "0",
            "label": "take_profit_first", "seconds_remaining": 300,
        })
    rows, quality = load_reversal_examples(str(path))
    assert rows == []
    assert quality["complete_reversal_markets"] == 0


def test_cycle_preserves_original_dbs_and_is_idempotent(tmp_path):
    research, outcomes, _ = _dbs(tmp_path)
    output = str(tmp_path / "ml")
    result = run_cycle(research, outcomes, output)
    assert result["status"] == "WAITING_FOR_MORE_SETTLED_MARKETS"
    assert result["data_quality"]["usable_settled_markets"] == 1
    assert result["models"]["outcome"]["logistic"]["status"] == "INSUFFICIENT_DATA"
    assert result["models"]["outcome"]["boosted_stumps"]["artifact"] is None
    assert result["automatic_promotion_enabled"] is False
    assert result["trading_enabled"] is False
    assert (tmp_path / "ml" / "latest_report.json").exists()
    assert (tmp_path / "ml" / "outcome_dataset.csv").exists()
    with open(tmp_path / "ml" / "latest_report.json") as fh:
        disk = json.load(fh)
    assert disk["data_quality"]["usable_settled_markets"] == 1
    next_run = run_cycle(research, outcomes, output)
    assert next_run["status"] == "NO_NEW_SETTLEMENT"
    with sqlite3.connect(outcomes) as con:
        assert con.execute("SELECT COUNT(*) FROM settled_outcomes").fetchone()[0] == 1
    with sqlite3.connect(research) as con:
        assert con.execute("SELECT COUNT(*) FROM research_markets").fetchone()[0] == 1
    with sqlite3.connect(str(tmp_path / "ml" / "ml_runs.db")) as con:
        assert con.execute("SELECT COUNT(*) FROM ml_research_runs").fetchone()[0] == 1
