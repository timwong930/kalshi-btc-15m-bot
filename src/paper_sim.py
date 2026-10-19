"""Phase 4: prospective virtual paper execution, NEVER calls an exchange.

Reads the append-only public collector DB in mode=ro; writes ONLY to its
own local SQLite ledger. Uses future quote events *as they arrive* for exits.
No demo orders, credentials, network calls, model promotions, or live trading.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import logging
import math
import os
import sqlite3
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

from src.ml_dataset import FEATURES, REVERSAL_FEATURES, _asof, _vol_60s, read_only

LOG = logging.getLogger("virtual_paper")
ONE_MINUTE_MS = 60_000
STRATEGIES = (
    "price_rule", "momentum_rule",
    "outcome_logistic", "outcome_boosted_stumps",
    "reversal_logistic", "reversal_boosted_stumps",
)
ALGORITHM_NAMES = {"logistic": "offline-logistic-v1", "boosted_stumps": "offline-boosted-stumps-v1"}


@dataclass(frozen=True)
class Policy:
    entry_window_start_seconds: int = 300
    entry_window_end_seconds: int = 240
    time_exit_seconds: int = 45
    max_quote_delay_ms: int = 25_000
    max_quote_gap_ms: int = 30_000
    min_depth: float = 5.0
    max_spread: float = 0.08
    fee_per_side: float = 0.02
    slippage_per_side: float = 0.01
    take_profit: float = 0.10
    stop_loss: float = 0.10
    min_hold_ms: int = 1_000
    model_outcome_edge: float = 0.08
    model_reversal_min_probability: float = 0.70

    def __post_init__(self):
        if not (0 < self.entry_window_end_seconds < self.entry_window_start_seconds < 900):
            raise ValueError("Invalid entry window")
        if not (0 < self.time_exit_seconds < self.entry_window_end_seconds):
            raise ValueError("Invalid exit window")
        values = (
            self.max_quote_delay_ms, self.max_quote_gap_ms, self.min_depth,
            self.max_spread, self.fee_per_side, self.slippage_per_side,
            self.take_profit, self.stop_loss, self.min_hold_ms,
            self.model_outcome_edge, self.model_reversal_min_probability,
        )
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Invalid paper trading policy")
        if self.max_quote_gap_ms < self.max_quote_delay_ms:
            raise ValueError("Quote gap must be >= freshness threshold")


def open_ledger(path: str, collector: sqlite3.Connection) -> sqlite3.Connection:
    """New ledgers begin at the collector's current max ID: NO historical fills."""
    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    is_new = not dest.exists()
    db = sqlite3.connect(str(dest), timeout=20)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS paper_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS paper_decisions(
            ticker TEXT NOT NULL, strategy TEXT NOT NULL,
            event_ms INTEGER NOT NULL, quote_id INTEGER NOT NULL,
            decision TEXT NOT NULL, reason TEXT NOT NULL,
            probability REAL, model_digest TEXT,
            PRIMARY KEY(ticker, strategy)
        );
        CREATE TABLE IF NOT EXISTS paper_trades(
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL, strategy TEXT NOT NULL,
            quote_source TEXT NOT NULL, side TEXT NOT NULL,
            opened_ms INTEGER NOT NULL, opened_quote_id INTEGER NOT NULL,
            close_ms INTEGER NOT NULL, entry_ask REAL NOT NULL,
            entry_slippage REAL NOT NULL, entry_fee REAL NOT NULL,
            entry_depth REAL NOT NULL, model_probability REAL, model_digest TEXT,
            status TEXT NOT NULL, exit_ms INTEGER, exit_quote_id INTEGER,
            exit_bid REAL, exit_slippage REAL, exit_fee REAL,
            pnl_per_contract REAL, reason TEXT,
            last_quote_ms INTEGER NOT NULL,
            UNIQUE(ticker, strategy)
        );
        CREATE INDEX IF NOT EXISTS idx_paper_open ON paper_trades(status, ticker);
    """)
    if is_new:
        cursor = collector.execute("SELECT COALESCE(MAX(id),0) FROM research_quotes").fetchone()[0]
        db.execute("INSERT INTO paper_meta VALUES('last_quote_id', ?)", (str(cursor),))
        db.execute("INSERT INTO paper_meta VALUES('starting_quote_id', ?)", (str(cursor),))
        db.execute("INSERT INTO paper_meta VALUES('created_epoch_ms', ?)", (str(int(time.time() * 1000)),))
        db.commit()
    elif db.execute("SELECT value FROM paper_meta WHERE key='last_quote_id'").fetchone() is None:
        db.close()
        raise ValueError("Existing paper DB has no cursor; refusing to infer or replay old trades")
    return db


def fresh(row: sqlite3.Row, now_ms: int, policy: Policy) -> bool:
    stamp = row["received_ms"]
    return 0 <= now_ms - stamp <= policy.max_quote_delay_ms


def quote_valid(row: sqlite3.Row, policy: Policy) -> bool:
    try:
        values = [row[key] for key in (
            "yes_bid", "yes_ask", "no_bid", "no_ask", "yes_depth", "no_depth"
        )]
        if not row["valid"] or any(v is None or not math.isfinite(v) for v in values):
            return False
        yb, ya, nb, na, yd, nd = values
        return (0 < yb <= ya < 1 and 0 < nb <= na < 1
                and ya - yb <= policy.max_spread
                and na - nb <= policy.max_spread
                and min(yd, nd) >= policy.min_depth)
    except (ValueError, TypeError):
        return False


def side_ask(row: sqlite3.Row, side: str) -> float:
    return float(row["yes_ask"] if side == "up" else row["no_ask"])


def side_bid(row: sqlite3.Row, side: str) -> float:
    return float(row["yes_bid"] if side == "up" else row["no_bid"])


def side_depth(row: sqlite3.Row, side: str) -> float:
    # YES ask comes from NO resting bids, and vice versa.
    return float(row["no_depth"] if side == "up" else row["yes_depth"])


def _predict_shadow(artifact: dict, values: tuple[float, ...]) -> float | None:
    try:
        if len(values) != len(artifact["feature_names"]) or not all(math.isfinite(v) for v in values):
            return None
        algo = artifact["algorithm"]
        if algo == ALGORITHM_NAMES["logistic"]:
            means, scales, weights = (
                artifact["feature_means"], artifact["feature_scales"], artifact["weights"]
            )
            if not (len(means) == len(scales) == len(weights) == len(values)):
                return None
            z = float(artifact["intercept"]) + sum(
                w * (value - mu) / scale
                for value, mu, scale, w in zip(values, means, scales, weights)
                if scale > 0
            )
            if any(scale <= 0 for scale in scales):
                return None
        elif algo == ALGORITHM_NAMES["boosted_stumps"]:
            z = float(artifact["intercept"])
            for j, threshold, left, right in artifact["trees"]:
                if not (0 <= int(j) < len(values)):
                    return None
                z += float(artifact["learning_rate"]) * (
                    float(left) if values[int(j)] <= threshold else float(right)
                )
        else:
            return None
        if not math.isfinite(z):
            return None
        result = 1 / (1 + math.exp(-max(-60, min(60, z))))
        return max(1e-7, min(1 - 1e-7, result))
    except (ValueError, OverflowError, TypeError, KeyError, IndexError, ZeroDivisionError):
        return None


def shadow_artifacts(directory: str, time_ms: int) -> dict[str, tuple[dict, str]]:
    """Freeze only artifact files known and completed strictly before entry."""
    output = {}
    base = Path(directory)
    if not base.is_dir():
        return output
    for path in base.glob("*.json"):
        try:
            stat = path.stat()
            if int(stat.st_mtime * 1000) >= time_ms:
                continue
            raw = path.read_bytes()
            model = json.loads(raw)
            task, algo = model["research_task"], model["algorithm"]
            if task not in ("outcome", "reversal") or algo not in ALGORITHM_NAMES.values():
                continue

            if model.get("mode") != "SHADOW_RESEARCH_ONLY" or model.get("live_trading_approval") is not False:
                continue
            if int(model["trained_through_close_ms"]) >= time_ms:
                continue
            expected = FEATURES if task == "outcome" else REVERSAL_FEATURES
            if tuple(model.get("feature_names", ())) != expected:
                continue
            algo_key = next(k for k, v in ALGORITHM_NAMES.items() if v == algo)
            key = f"{task}_{algo_key}"
            # Favor newest eligible artifact for each strategy, never an updated file from the future.
            if key not in output or stat.st_mtime > output[key][2]:
                output[key] = (model, hashlib.sha256(raw).hexdigest(), stat.st_mtime)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    return {key: (value[0], value[1]) for key, value in output.items()}


def market_context(research: sqlite3.Connection, row: sqlite3.Row):
    ticker = row["ticker"]
    return research.execute(
        "SELECT ticker, close_ms, strike, strike_type FROM research_markets WHERE ticker=?",
        (ticker,),
    ).fetchone()


def causal_features(research: sqlite3.Connection, row: sqlite3.Row, market: sqlite3.Row):
    """No outcome lookup, and no received timestamps later than the quote."""
    timestamp = row["received_ms"]
    coinbase = _asof(research, "coinbase", timestamp)
    brti = _asof(research, "brti", timestamp)
    spot = brti if brti is not None else coinbase
    strike = market["strike"]
    if market["strike_type"] != "greater_or_equal" or not strike or not spot:
        return None
    vol = _vol_60s(research, timestamp)
    if vol is None:
        return None
    spot_bps = max(-200, min(200, (spot - strike) / spot * 10000))
    fraction = (market["close_ms"] - timestamp) / 900000
    outcome = (
        (row["yes_bid"] + row["yes_ask"]) / 2, spot_bps, fraction,
        row["yes_ask"] - row["yes_bid"],
        math.log1p(row["yes_depth"]), math.log1p(row["no_depth"]),
        vol, float(brti is not None),
    )
    reversal = {
        side: (side_ask(row, side), side_ask(row, side) - side_bid(row, side),
               math.log1p(side_depth(row, side)), fraction,
               spot_bps * (1 if side == "up" else -1), float(side == "up"))
        for side in ("up", "down")
    }
    return outcome, reversal


def momentum_side(research: sqlite3.Connection, now_ms: int) -> str | None:
    current = _asof(research, "coinbase", now_ms, max_age_ms=10000)
    earlier = _asof(research, "coinbase", now_ms - ONE_MINUTE_MS, max_age_ms=10000)
    if current is None or earlier is None:
        return None
    change = current / earlier - 1
    return "up" if change >= 0.00025 else "down" if change <= -0.00025 else None


def choose_side(
    strategy: str, research: sqlite3.Connection, row: sqlite3.Row,
    features, artifacts: dict, policy: Policy,
) -> tuple[str | None, float | None, str | None, str]:
    asks = {"up": side_ask(row, "up"), "down": side_ask(row, "down")}
    if strategy == "price_rule":
        side = min(asks, key=asks.get)
        if not 0.15 <= asks[side] <= 0.35:
            return None, None, None, "price_not_in_fixed_range"
        return side, None, None, "fixed_price_rule"
    if strategy == "momentum_rule":
        side = momentum_side(research, row["received_ms"])
        if side is None or asks[side] > 0.75:
            return None, None, None, "no_causal_momentum_signal"
        return side, None, None, "fixed_momentum_rule"
    key = strategy
    package = artifacts.get(key)
    if package is None:
        return None, None, None, "no_eligible_preentry_shadow_artifact"
    model, digest = package
    if features is None:
        return None, None, digest, "missing_causal_model_features"
    outcome_features, reversal_features = features
    if strategy.startswith("outcome_"):
        p = _predict_shadow(model, outcome_features)
        if p is None:
            return None, None, digest, "invalid_outcome_model"
        scores = {
            "up": p - asks["up"] - policy.fee_per_side - policy.slippage_per_side,
            "down": 1 - p - asks["down"] - policy.fee_per_side - policy.slippage_per_side,
        }
        side = max(scores, key=scores.get)
        if scores[side] < policy.model_outcome_edge:
            return None, p, digest, "no_modeled_net_settlement_edge"
        return side, p, digest, "shadow_outcome_edge"
    scored = {side: _predict_shadow(model, vals) for side, vals in reversal_features.items()}
    scored = {k: v for k, v in scored.items() if v is not None and asks[k] <= 0.80}
    if not scored:
        return None, None, digest, "invalid_reversal_model_or_ask"
    side = max(scored, key=scored.get)
    p = scored[side]
    if p < policy.model_reversal_min_probability:
        return None, p, digest, "reversal_probability_below_threshold"
    return side, p, digest, "shadow_reversal_probability"


def _record_decision(
    ledger: sqlite3.Connection, row: sqlite3.Row, strategy: str, side: str | None,
    probability: float | None, digest: str | None, reason: str,
    market: sqlite3.Row, policy: Policy,
) -> None:
    ledger.execute(
        "INSERT OR IGNORE INTO paper_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (row["ticker"], strategy, row["received_ms"], row["id"],
         "ENTRY" if side else "SKIP", reason, probability, digest),
    )
    if side:
        ledger.execute(
            """INSERT OR IGNORE INTO paper_trades
               (ticker, strategy, quote_source, side, opened_ms, opened_quote_id,
                close_ms, entry_ask, entry_slippage, entry_fee, entry_depth,
                model_probability, model_digest, status, last_quote_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)""",
            (row["ticker"], strategy, row["source"], side,
             row["received_ms"], row["id"], market["close_ms"], side_ask(row, side),
             policy.slippage_per_side, policy.fee_per_side, side_depth(row, side),
             probability, digest, row["received_ms"]),
        )


def _exit_trade(
    ledger: sqlite3.Connection, trade: sqlite3.Row, row: sqlite3.Row,
    policy: Policy, reason: str,
) -> None:
    bid = side_bid(row, trade["side"])
    pnl = (bid - policy.slippage_per_side - policy.fee_per_side
           - trade["entry_ask"] - trade["entry_slippage"] - trade["entry_fee"])
    ledger.execute(
        """UPDATE paper_trades SET status='CLOSED', exit_ms=?, exit_quote_id=?,
           exit_bid=?, exit_slippage=?, exit_fee=?, pnl_per_contract=?, reason=?,
           last_quote_ms=? WHERE id=? AND status='OPEN'""",
        (row["received_ms"], row["id"], bid, policy.slippage_per_side,
         policy.fee_per_side, pnl, reason, row["received_ms"], trade["id"]),
    )


def _censor(ledger: sqlite3.Connection, trade: sqlite3.Row, reason: str, timestamp: int) -> None:
    ledger.execute(
        """UPDATE paper_trades SET status='CENSORED', reason=?, last_quote_ms=?
           WHERE id=? AND status='OPEN'""", (reason, timestamp, trade["id"]),
    )


def process_quote(
    research: sqlite3.Connection, ledger: sqlite3.Connection,
    row: sqlite3.Row, now_ms: int, policy: Policy, model_dir: str,
) -> None:
    """One strictly ordered collector event. Never synthesize a successful fill."""
    market = market_context(research, row)
    if market is None:
        return
    timestamp, close_ms = row["received_ms"], market["close_ms"]
    timely = fresh(row, now_ms, policy)
    # No cursor rewind: stale data after downtime cannot create new entries
    # or retroactively book profitable exits.
    for trade in ledger.execute(
        "SELECT * FROM paper_trades WHERE ticker=? AND status='OPEN'",
        (row["ticker"],),
    ).fetchall():
        if row["source"] != trade["quote_source"] or row["id"] <= trade["opened_quote_id"]:
            continue
        if timestamp <= trade["last_quote_ms"]:
            continue
        if timestamp - trade["last_quote_ms"] > policy.max_quote_gap_ms or not timely:
            _censor(ledger, trade, "missing_or_delayed_exit_quotes", timestamp)
            continue
        if not quote_valid(row, policy):
            _censor(ledger, trade, "invalid_exit_book_or_depth", timestamp)
            continue
        if timestamp > close_ms or timestamp < trade["opened_ms"] + policy.min_hold_ms:
            continue
        pnl = (side_bid(row, trade["side"]) - policy.slippage_per_side - policy.fee_per_side
               - trade["entry_ask"] - trade["entry_fee"] - trade["entry_slippage"])
        if pnl >= policy.take_profit:
            reason = "take_profit"
        elif pnl <= -policy.stop_loss:
            reason = "stop_loss"
        elif close_ms - timestamp <= policy.time_exit_seconds * 1000:
            reason = "preclose_time_exit"
        else:
            ledger.execute(
                "UPDATE paper_trades SET last_quote_ms=? WHERE id=?", (timestamp, trade["id"])
            )
            continue
        _exit_trade(ledger, trade, row, policy, reason)

    remaining = (close_ms - timestamp) / 1000
    if not (policy.entry_window_end_seconds <= remaining <= policy.entry_window_start_seconds):
        return
    # Even SKIP decisions are recorded once; no repeated attempts later
    # with hindsight or improved quote selection.
    if not timely or not quote_valid(row, policy):
        return
    existing = {
        item["strategy"] for item in ledger.execute(
            "SELECT strategy FROM paper_decisions WHERE ticker=?", (row["ticker"],)
        )
    }
    if len(existing) == len(STRATEGIES):
        return
    models = shadow_artifacts(model_dir, timestamp)
    features = causal_features(research, row, market)
    for strategy in STRATEGIES:
        if strategy in existing:
            continue
        side, p, digest, reason = choose_side(strategy, research, row, features, models, policy)
        _record_decision(ledger, row, strategy, side, p, digest, reason, market, policy)


def expire_unclosed(ledger: sqlite3.Connection, now_ms: int) -> None:
    """Don't invent a close trade or a settlement where quotes are missing."""
    ledger.execute(
        """UPDATE paper_trades
           SET status='CENSORED', reason='no_executable_preclose_quote'
           WHERE status='OPEN' AND close_ms < ?""", (now_ms - 5000,),
    )


def run_tick(
    collector: sqlite3.Connection, ledger: sqlite3.Connection, model_dir: str,
    policy: Policy, now_ms: int | None = None,
) -> dict:
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    cursor = int(ledger.execute(
        "SELECT value FROM paper_meta WHERE key='last_quote_id'"
    ).fetchone()[0])
    # Read-only collector supports concurrent WAL writes; do not touch it.
    incoming = collector.execute(
        "SELECT * FROM research_quotes WHERE id>? ORDER BY id LIMIT 1000", (cursor,)
    ).fetchall()
    for row in incoming:
        # Transaction per received row means restarts cannot double-count.
        with ledger:
            process_quote(collector, ledger, row, now_ms, policy, model_dir)
            ledger.execute(
                "UPDATE paper_meta SET value=? WHERE key='last_quote_id'", (str(row["id"]),)
            )
    with ledger:
        expire_unclosed(ledger, now_ms)
    return {"processed": len(incoming), "last_quote_id": incoming[-1]["id"] if incoming else cursor}


def report(ledger: sqlite3.Connection) -> dict:
    from statistics import mean

    metrics = {}
    for name in STRATEGIES:
        trades = ledger.execute(
            "SELECT * FROM paper_trades WHERE strategy=? ORDER BY exit_ms, id", (name,)
        ).fetchall()
        closed = [t for t in trades if t["status"] == "CLOSED"]
        equity, peak, drawdown = 0.0, 0.0, 0.0
        for trade in closed:
            equity += trade["pnl_per_contract"]
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        wins = sum(t["pnl_per_contract"] > 0 for t in closed)
        losses = [t["pnl_per_contract"] for t in closed if t["pnl_per_contract"] < 0]
        gains = [t["pnl_per_contract"] for t in closed if t["pnl_per_contract"] > 0]
        decisions = ledger.execute(
            "SELECT COUNT(*) FROM paper_decisions WHERE strategy=?", (name,)
        ).fetchone()[0]
        metrics[name] = {
            "decisions": decisions,
            "closed_trades": len(closed),
            "open_trades": sum(t["status"] == "OPEN" for t in trades),
            "censored_trades": sum(t["status"] == "CENSORED" for t in trades),
            "wins": wins,
            "win_rate": wins / len(closed) if closed else None,
            "net_simulated_pnl_per_one_contract_each": round(equity, 6),
            "mean_simulated_pnl_per_closed_trade": mean(t["pnl_per_contract"] for t in closed) if closed else None,
            "max_closed_trade_drawdown_dollars": round(drawdown, 6),
            "gross_profit": round(sum(gains), 6),
            "gross_loss": round(sum(losses), 6),
            "profit_factor": sum(gains) / -sum(losses) if losses else None,
            "note": "Censored and open trades excluded from P&L; compare rates and selection bias",
        }
    return {
        "mode": "VIRTUAL_QUOTES_ONLY_NOT_EXCHANGE_PAPER_ORDERS",
        "never_actual_fills": True,
        "trading_enabled": False,
        "fee_slippage_scenarios_not_verified_exchange_fees": True,
        "no_trade_baseline_pnl": 0.0,
        "strategies": metrics,
    }


def _atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(contents, encoding="utf-8")
    os.replace(pending, path)


def save_reports(ledger: sqlite3.Connection, folder: str) -> dict:
    """Export persistent per-trade evidence and closed-trade equity histories."""
    import io

    output = Path(folder).resolve()
    output.mkdir(parents=True, exist_ok=True)
    totals = report(ledger)
    totals["reported_epoch_ms"] = int(time.time() * 1000)
    _atomic_text(output / "latest_metrics.json", json.dumps(totals, sort_keys=True, indent=2) + "\n")
    trades = ledger.execute(
        "SELECT * FROM paper_trades ORDER BY strategy, opened_ms, id"
    ).fetchall()
    buff = io.StringIO()
    if trades:
        fields = list(trades[0].keys())
        writer = csv.DictWriter(buff, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(t) for t in trades)
    else:
        buff.write("id,ticker,strategy,status,pnl_per_contract\n")
    _atomic_text(output / "paper_trades.csv", buff.getvalue())
    curve = io.StringIO()
    writer = csv.writer(curve)
    writer.writerow(["strategy", "exit_ms", "ticker", "trade_id", "pnl_per_contract", "cumulative_pnl", "drawdown"])
    for strategy in STRATEGIES:
        prior = ledger.execute(
            "SELECT id, ticker, exit_ms, pnl_per_contract FROM paper_trades "
            "WHERE strategy=? AND status='CLOSED' ORDER BY exit_ms, id",
            (strategy,),
        ).fetchall()
        equity = peak = 0.0
        for trade in prior:
            equity += trade["pnl_per_contract"]
            peak = max(peak, equity)
            writer.writerow([
                strategy, trade["exit_ms"], trade["ticker"], trade["id"],
                trade["pnl_per_contract"], round(equity, 6), round(peak - equity, 6),
            ])
    _atomic_text(output / "equity_curves.csv", curve.getvalue())
    return totals


@contextmanager
def singleton(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another virtual paper process is running") from exc
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only prospective VIRTUAL paper trading; never submits orders")
    parser.add_argument("--research-db", help="Read-only collector database; required unless --status")
    parser.add_argument("--paper-db", default="data/paper_research/paper.db")
    parser.add_argument("--reports-dir", default="data/paper_research")
    parser.add_argument("--model-dir", default="data/ml_research/shadow_models")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--report-seconds", type=float, default=300)
    args = parser.parse_args()
    if args.report_seconds < 10:
        parser.error("Report interval must be >=10 seconds")
    if args.poll_seconds < 1:
        parser.error("Poll interval must be >=1 second")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policy = Policy()
    if not args.research_db and not args.status:
        parser.error("--research-db is required to run the prospective simulator")
    if args.status:
        if not Path(args.paper_db).is_file():
            print("No virtual paper ledger exists yet")
            return
        with closing(sqlite3.connect(args.paper_db)) as db:
            db.row_factory = sqlite3.Row
            print(json.dumps(save_reports(db, args.reports_dir), indent=2))
        return
    with singleton(Path(args.paper_db).resolve().parent / "paper.lock"):
        with closing(read_only(args.research_db)) as source:
            with closing(open_ledger(args.paper_db, source)) as db:
                last_report = 0.0
                while True:
                    counters = run_tick(source, db, args.model_dir, policy)
                    if counters["processed"]:
                        LOG.info(
                            "Processed %s quote events; cursor=%s",
                            counters["processed"], counters["last_quote_id"],
                        )
                    if not args.loop or time.monotonic() - last_report >= args.report_seconds:
                        snapshot = save_reports(db, args.reports_dir)
                        last_report = time.monotonic()
                        LOG.info("Virtual paper report saved: %s closed trades",
                                 sum(item["closed_trades"] for item in snapshot["strategies"].values()))
                    if not args.loop:
                        print(json.dumps(snapshot, indent=2))
                        break
                    time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
