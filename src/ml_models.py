"""Phase 3: deterministic, dependency-free logistic ML challenger + walk-forward audit.

No live inference, order routing, auto-promotion, or retraining existing bot.
Fit scalers only on each training fold; evaluate on later unseen MARKETS.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, pstdev

from src.ml_dataset import Example

MODEL_VERSION = "offline-logistic-v1"
EPS = 1e-7


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-min(value, 60)))
    exp_value = math.exp(max(value, -60))
    return exp_value / (1 + exp_value)


def _clamp(value: float) -> float:
    return max(EPS, min(1 - EPS, value))


@dataclass(frozen=True)
class LogisticModel:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    intercept: float

    def predict(self, features: tuple[float, ...]) -> float:
        standard = [(x - m) / s for x, m, s in zip(features, self.mean, self.scale)]
        score = self.intercept + sum(w * x for w, x in zip(self.weights, standard))
        return _clamp(_sigmoid(score))

    def as_dict(self) -> dict:
        return {
            "algorithm": MODEL_VERSION, "feature_means": list(self.mean),
            "feature_scales": list(self.scale), "weights": list(self.weights),
            "intercept": self.intercept,
        }


def train_logistic(examples: list[Example], epochs: int = 300) -> LogisticModel:
    if len(examples) < 2:
        raise ValueError("Not enough training rows")
    dims = len(examples[0].features)
    if dims < 1 or any(len(e.features) != dims for e in examples):
        raise ValueError("Feature dimensionality mismatch")
    if any(not all(math.isfinite(v) for v in e.features) for e in examples):
        raise ValueError("Nonfinite training feature")
    means = tuple(mean(e.features[j] for e in examples) for j in range(dims))
    scales = tuple(max(1e-5, pstdev([e.features[j] for e in examples])) for j in range(dims))
    inputs = [
        [(v - m) / s for v, m, s in zip(e.features, means, scales)]
        for e in examples
    ]
    # Deterministic full-batch training, L2 penalty on weights, not intercept.
    weights = [0.0] * dims
    prior = _clamp(mean(e.y for e in examples))
    bias = math.log(prior / (1 - prior))
    count = len(examples)
    for _ in range(epochs):
        gradients = [0.0] * dims
        intercept_grad = 0.0
        for e, x in zip(examples, inputs):
            p = _sigmoid(bias + sum(a * b for a, b in zip(weights, x)))
            error = p - e.y
            intercept_grad += error
            for j, v in enumerate(x):
                gradients[j] += error * v
        step = 0.10
        for j in range(dims):
            weights[j] -= step * (gradients[j] / count + 0.005 * weights[j])
        bias -= step * intercept_grad / count
    return LogisticModel(means, scales, tuple(weights), bias)


@dataclass(frozen=True)
class BoostedStumps:
    intercept: float
    trees: tuple[tuple[int, float, float, float], ...]
    learning_rate: float = 0.10

    def predict(self, features: tuple[float, ...]) -> float:
        z = self.intercept
        for feature, threshold, left, right in self.trees:
            z += self.learning_rate * (left if features[feature] <= threshold else right)
        return _clamp(_sigmoid(z))

    def as_dict(self) -> dict:
        return {
            "algorithm": "offline-boosted-stumps-v1",
            "intercept": self.intercept,
            "learning_rate": self.learning_rate,
            "trees": [list(tree) for tree in self.trees],
        }


def train_boosted_stumps(examples: list[Example], rounds: int = 40) -> BoostedStumps:
    """Small regularized gradient-boosted classification trees (depth=1).

    All split thresholds come only from training observations. This is a
    deliberately conservative, dependency-free alternative to large GBMs.
    """
    if len(examples) < 2:
        raise ValueError("Not enough training rows")
    dimensions = len(examples[0].features)
    if dimensions < 1 or any(len(row.features) != dimensions for row in examples):
        raise ValueError("Inconsistent training dimensions")
    prior = _clamp(mean(row.y for row in examples))
    intercept = math.log(prior / (1 - prior))
    predictions = [prior] * len(examples)
    trees = []
    candidates = []
    for j in range(dimensions):
        values = sorted({e.features[j] for e in examples})
        if len(values) < 2:
            candidates.append([])
            continue
        # At most four distinct split proposals per feature; no test leakage.
        thresholds = sorted({
            (values[max(0, min(len(values) - 2, int(q * (len(values) - 1))))] +
             values[max(1, min(len(values) - 1, int(q * (len(values) - 1)) + 1))]) / 2
            for q in (0.2, 0.4, 0.6, 0.8)
        })
        candidates.append(thresholds)
    for _ in range(rounds):
        residuals = [e.y - p for e, p in zip(examples, predictions)]
        hessians = [max(0.02, p * (1 - p)) for p in predictions]
        best = None
        for j, thresholds in enumerate(candidates):
            for threshold in thresholds:
                lg = lh = rg = rh = 0.0
                ln = rn = 0
                for row, g, h in zip(examples, residuals, hessians):
                    if row.features[j] <= threshold:
                        lg += g
                        lh += h
                        ln += 1
                    else:
                        rg += g
                        rh += h
                        rn += 1
                if min(ln, rn) < max(3, len(examples) // 20):
                    continue
                gain = lg * lg / (lh + 5) + rg * rg / (rh + 5)
                if best is None or gain > best[0]:
                    best = (gain, j, threshold, lg / (lh + 5), rg / (rh + 5))
        if best is None or best[0] < 1e-8:
            break
        _, feature, threshold, left, right = best
        left, right = max(-1, min(1, left)), max(-1, min(1, right))
        trees.append((feature, threshold, left, right))
        predictions = [
            _sigmoid(math.log(_clamp(p) / (1 - _clamp(p)))
                     + 0.10 * (left if row.features[feature] <= threshold else right))
            for row, p in zip(examples, predictions)
        ]
    return BoostedStumps(intercept, tuple(trees))


def fit_model(examples: list[Example], algorithm: str):
    if algorithm == "logistic":
        return train_logistic(examples)
    if algorithm == "boosted_stumps":
        return train_boosted_stumps(examples)
    raise ValueError(f"Unsupported shadow model: {algorithm}")


def brier(p: float, label: int) -> float:
    return (_clamp(p) - label) ** 2


def log_loss(p: float, label: int) -> float:
    safe = _clamp(p)
    return -(label * math.log(safe) + (1 - label) * math.log(1 - safe))


def grouped_markets(examples: list[Example]) -> list[tuple[int, str, list[Example]]]:
    groups = defaultdict(list)
    for e in examples:
        groups[(e.close_ms, e.ticker)].append(e)
    return [(close, ticker, group) for (close, ticker), group in sorted(groups.items())]


def walk_forward(
    examples: list[Example],
    min_train_markets: int = 60,
    fold_test_markets: int = 20,
    *,
    baseline_type: str = "market",
    algorithm: str = "logistic",
) -> dict:
    """Expanding chronological folds, with paired uncertainty BY MARKET.

    Fold construction ensures no market from the test cohort is in train,
    and all training closes precede each testing entry.
    """
    groups = grouped_markets(examples)
    result: dict = {
        "model": algorithm, "total_rows": len(examples),
        "distinct_markets": len(groups),
        "folds": 0, "tested_markets": 0,
        "status": "INSUFFICIENT_DATA",
        "min_train_markets": min_train_markets,
        "fold_test_markets": fold_test_markets,
    }
    if min_train_markets < 2 or fold_test_markets < 1:
        raise ValueError("Invalid walk-forward fold sizing")
    if len(groups) < min_train_markets + fold_test_markets:
        result["markets_needed_for_first_fold"] = max(
            0, min_train_markets + fold_test_markets - len(groups)
        )
        return result
    baseline_losses, model_losses, base_logs, model_logs = [], [], [], []
    differences = []
    fold_summaries = []
    for end in range(min_train_markets, len(groups) - fold_test_markets + 1, fold_test_markets):
        train_groups = groups[:end]
        test_groups = groups[end:end + fold_test_markets]
        train = [r for _, _, cohort in train_groups for r in cohort]
        if not train or not test_groups:
            continue
        # Reject chronology violations rather than silently leaking outcomes.
        if max(r.close_ms for r in train) >= min(
            r.entry_ms for _, _, cohort in test_groups for r in cohort
        ):
            continue
        model = fit_model(train, algorithm)
        train_rate = _clamp(mean(row.y for row in train))
        fold_market_differences = []
        for _, _, cohort in test_groups:
            observed_model = [brier(model.predict(e.features), e.y) for e in cohort]
            observed_base = [
                brier(e.baseline if baseline_type == "market" else train_rate, e.y)
                for e in cohort
            ]
            log_model = [log_loss(model.predict(e.features), e.y) for e in cohort]
            log_base = [
                log_loss(e.baseline if baseline_type == "market" else train_rate, e.y)
                for e in cohort
            ]
            model_score, base_score = mean(observed_model), mean(observed_base)
            model_losses.append(model_score)
            baseline_losses.append(base_score)
            model_logs.append(mean(log_model))
            base_logs.append(mean(log_base))
            difference = base_score - model_score
            fold_market_differences.append(difference)
            differences.append(difference)
        fold_summaries.append({
            "train_markets": len(train_groups),
            "test_markets": len(test_groups),
            "train_last_close_ms": train_groups[-1][0],
            "test_first_entry_ms": min(
                r.entry_ms for _, _, cohort in test_groups for r in cohort
            ),
            "brier_improvement": mean(fold_market_differences),
        })
    if not differences:
        result["status"] = "NO_VALID_CHRONOLOGICAL_FOLDS"
        return result
    difference = mean(differences)
    margin = 1.96 * pstdev(differences) / math.sqrt(len(differences))
    result.update({
        "status": "SHADOW_EVALUATED",
        "folds": len(fold_summaries),
        "tested_markets": len(differences),
        "market_baseline_brier": mean(baseline_losses),
        "challenger_brier": mean(model_losses),
        "market_baseline_log_loss": mean(base_logs),
        "challenger_log_loss": mean(model_logs),
        "paired_brier_improvement": difference,
        "paired_brier_95pct_interval": [difference - margin, difference + margin],
        "brier_improvement_supported": len(differences) >= 100 and difference - margin > 0,
        "fold_details": fold_summaries,
        "approval": "RESEARCH_ONLY_NEVER_AUTO_PROMOTE",
    })
    return result


def final_shadow_model(
    examples: list[Example], feature_names: tuple[str, ...], algorithm: str = "logistic"
) -> dict:
    """Serialized artifact for offline inspection ONLY, not consumed by the bot."""
    if len(examples[0].features) != len(feature_names):
        raise ValueError("Artifact feature names do not match examples")
    model = fit_model(examples, algorithm)
    return {
        **model.as_dict(),
        "feature_names": list(feature_names),
        "trained_through_close_ms": max(x.close_ms for x in examples),
        "training_market_count": len({x.ticker for x in examples}),
        "training_rows": len(examples),
        "mode": "SHADOW_RESEARCH_ONLY",
        "live_trading_approval": False,
    }
