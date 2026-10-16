"""Phase 2+3 autonomous research loop (no network, orders or live promotion).

Reads two existing SQLite databases strictly READ ONLY, builds settled-only
datasets, performs expanding walk-forward tests by market, and writes shadow
artifacts to a third, separate research directory.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.ml_dataset import (
    FEATURES,
    REVERSAL_FEATURES,
    collect_outcome_examples,
    load_reversal_examples,
    read_only,
    stream_quality,
    write_examples,
)
from src.ml_models import final_shadow_model, walk_forward
from src.research_labels import build_labels

LOG = logging.getLogger("kalshi_ml_research")
ALGORITHMS = ("logistic", "boosted_stumps")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    os.replace(temporary, path)


@contextmanager
def singleton(lock_path: Path):
    """Non-blocking exclusive lock: cannot run two retrainers on one registry."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another ML trainer instance holds the lock") from exc
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def initialize_registry(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=20)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ml_research_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_utc TEXT NOT NULL,
            dataset_hash TEXT NOT NULL,
            newly_evaluated INTEGER NOT NULL,
            settled_markets INTEGER NOT NULL,
            usable_markets INTEGER NOT NULL,
            status TEXT NOT NULL,
            report_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ml_runs_created ON ml_research_runs(created_utc);
    """)
    conn.commit()
    return conn


def dataset_signature(research_path: str, outcomes_path: str) -> tuple[str, int]:
    """Changes only when a research market acquires/changes a settled result."""
    with closing(read_only(research_path)) as research, closing(read_only(outcomes_path)) as outcomes:
        markets = {r["ticker"]: r["close_ms"] for r in research.execute(
            "SELECT ticker, close_ms FROM research_markets"
        )}
        states = sorted(
            (ticker, markets[ticker], row["result"])
            for row in outcomes.execute(
                "SELECT ticker, result FROM settled_outcomes WHERE result IN ('yes','no')"
            )
            if (ticker := row["ticker"]) in markets
        )
        digest = hashlib.sha256(json.dumps(states, separators=(",", ":")).encode()).hexdigest()
        return digest, len(states)


def previous_run(registry: sqlite3.Connection) -> tuple[str | None, float]:
    row = registry.execute(
        "SELECT dataset_hash FROM ml_research_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    # Models saved as immutable artifacts have their own clock, not the cadence
    # of incomplete or no-new-data summary rows.
    model_time = registry.execute(
        "SELECT created_utc FROM ml_research_runs "
        "WHERE status='SHADOW_TRAINED' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    latest_ts = datetime.fromisoformat(model_time[0]).timestamp() if model_time else 0.0
    return (row[0] if row else None), latest_ts


def _atomic_dataset(path: Path, examples, names) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write_examples(examples, str(temporary), names)
    os.replace(temporary, path)


def run_cycle(
    research_db: str, outcomes_db: str, output_dir: str,
    *, min_train_markets: int = 60, fold_test_markets: int = 20,
    train_every_hours: float = 24, force: bool = False,
) -> dict:
    """Idempotent, autonomous settled-history preparation and shadow research."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    signature, settled_count = dataset_signature(research_db, outcomes_db)
    registry = initialize_registry(output / "ml_runs.db")
    try:
        last_signature, last_trained = previous_run(registry)
        new_settlements = signature != last_signature
        if not (new_settlements or force):
            return {
                "status": "NO_NEW_SETTLEMENT",
                "settled_research_markets": settled_count,
                "new_data": False,
                "report": str(output / "latest_report.json"),
            }

        outcomes, quality = collect_outcome_examples(research_db, outcomes_db)
        quality["stream_counts"] = stream_quality(research_db)
        quality["settled_research_markets"] = settled_count
        _atomic_dataset(output / "outcome_dataset.csv", outcomes, FEATURES)

        # Existing replay generator uses future quotes ONLY for the label,
        # never for the preceding feature vector. Incomplete windows are
        # explicitly excluded from the supervised reversal classifier.
        reversal_tmp = output / "reversal_labels.csv.tmp"
        reversal_count, complete_count = build_labels(
            research_db, outcomes_db, str(reversal_tmp),
        )
        os.replace(reversal_tmp, output / "reversal_labels.csv")
        reversals, reversal_quality = load_reversal_examples(str(output / "reversal_labels.csv"))
        quality.update(reversal_quality)
        quality.update({
            "reversal_candidate_rows": reversal_count,
            "reversal_complete_quote_rows": complete_count,
            "reversal_incomplete_or_unusable_rows": reversal_count - complete_count,
        })
        _atomic_dataset(output / "reversal_dataset.csv", reversals, REVERSAL_FEATURES)

        report: dict = {
            "generated_at_utc": utcnow(),
            "dataset_hash": signature,
            "data_quality": quality,
            "models": {},
            "mode": "READ_ONLY_SHADOW_RESEARCH",
            "trading_enabled": False,
            "automatic_promotion_enabled": False,
            "historical_market_count_not_independent_ticks": len(outcomes),
            "assumptions": {
                "reversal_fee_per_side_dollars": 0.02,
                "reversal_slippage_per_side_dollars": 0.01,
                "fill_simulation": "hypothetical top-of-book, no observed fills",
            },
        }
        next_allowed = last_trained + train_every_hours * 3600
        in_cooldown = time.time() < next_allowed and not force
        any_trained = False
        for task, examples, names, baseline_type in (
            ("outcome", outcomes, FEATURES, "market"),
            ("reversal", reversals, REVERSAL_FEATURES, "prevalence"),
        ):
            report["models"][task] = {}
            for algorithm in ALGORITHMS:
                if in_cooldown:
                    metrics = {
                        "status": "TRAINING_COOLDOWN", "tested_markets": 0,
                        "distinct_markets": len({e.ticker for e in examples}),
                        "model": algorithm,
                        "note": "New settled labels available; full walk-forward training runs daily",
                    }
                else:
                    metrics = walk_forward(
                        examples, min_train_markets, fold_test_markets,
                        baseline_type=baseline_type, algorithm=algorithm,
                    )
                metrics["artifact"] = None
                metrics["baseline"] = "Kalshi yes midpoint" if task == "outcome" else "training-only event prevalence"
                if metrics["status"] == "SHADOW_EVALUATED":
                    artifact = final_shadow_model(examples, names, algorithm)
                    artifact["dataset_hash"] = signature
                    artifact["research_task"] = task
                    name = f"{task}_{algorithm}_{signature[:12]}.json"
                    path = output / "shadow_models" / name
                    atomic_json(path, artifact)
                    metrics["artifact"] = str(path)
                    metrics["status"] = "SHADOW_TRAINED"
                    any_trained = True
                report["models"][task][algorithm] = metrics

        report["status"] = (
            "SHADOW_TRAINED" if any_trained else
            "LABELS_UPDATED_TRAINING_COOLDOWN" if in_cooldown else
            "WAITING_FOR_MORE_SETTLED_MARKETS"
        )
        atomic_json(output / "latest_report.json", report)
        registry.execute(
            "INSERT INTO ml_research_runs (created_utc, dataset_hash, newly_evaluated, "
            "settled_markets, usable_markets, status, report_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (report["generated_at_utc"], signature, int(new_settlements),
             settled_count, len(outcomes), report["status"], json.dumps(report, sort_keys=True)),
        )
        registry.commit()
        return report
    finally:
        registry.close()


def print_status(output_dir: str) -> None:
    output = Path(output_dir) / "latest_report.json"
    if not output.exists():
        print(f"Waiting for first ML research run: {output}")
        return
    with output.open(encoding="utf-8") as fh:
        report = json.load(fh)
    print(f"Status: {report['status']}, updated: {report['generated_at_utc']}")
    q = report["data_quality"]
    print(f"Research markets settled: {q['settled_research_markets']}")
    print(f"Usable outcome markets: {q['usable_settled_markets']}")
    print(f"Complete reversal markets: {q['complete_reversal_markets']}")
    print(f"Excluded: {q['excluded_market_reasons']}")
    for task, algorithms in report["models"].items():
        for algo, metrics in algorithms.items():
            extra = ""
            if metrics.get("tested_markets"):
                extra = (
                    f" challenger Brier={metrics['challenger_brier']:.5f}, "
                    f"baseline={metrics['market_baseline_brier']:.5f}"
                )
            print(f"{task}/{algo}: {metrics['status']}, "
                  f"tested markets={metrics['tested_markets']}{extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent Phase 2/3 shadow ML trainer; NO ORDERS")
    parser.add_argument("--research-db", default="data/research_stream.db")
    parser.add_argument("--outcomes-db", required=False, default="../kalshi-btc-15m-bot/data/kalshi_btc15m.db")
    parser.add_argument("--output-dir", default="data/ml_research")
    parser.add_argument("--min-train-markets", type=int, default=60)
    parser.add_argument("--fold-test-markets", type=int, default=20)
    parser.add_argument("--train-every-hours", type=float, default=24)
    parser.add_argument("--interval-minutes", type=float, default=15)
    parser.add_argument("--loop", action="store_true", help="Keep checking for newly settled markets")
    parser.add_argument("--status", action="store_true", help="Print latest research report and exit")
    parser.add_argument("--force", action="store_true", help="Recalculate even without new settlements")
    args = parser.parse_args()
    if args.status:
        print_status(args.output_dir)
        return
    if args.min_train_markets < 20 or args.fold_test_markets < 10:
        parser.error("Production research requires >=20 training and >=10 test markets per fold")
    if args.interval_minutes < 1 or args.train_every_hours < 0:
        parser.error("Invalid training cadence")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with singleton(Path(args.output_dir).resolve() / "trainer.lock"):
        while True:
            try:
                result = run_cycle(
                    args.research_db, args.outcomes_db, args.output_dir,
                    min_train_markets=args.min_train_markets,
                    fold_test_markets=args.fold_test_markets,
                    train_every_hours=args.train_every_hours, force=args.force,
                )
                LOG.info("ML research cycle: %s; %s settled markets",
                         result["status"], result.get("data_quality", {}).get(
                             "settled_research_markets", result.get("settled_research_markets", 0)))
            except Exception:
                LOG.exception("ML research cycle failed; previous artifacts preserved")
                if not args.loop:
                    raise
            if not args.loop:
                break
            time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()
