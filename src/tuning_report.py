"""Compare completed training-fold tuning runs and scan local model blends."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from src.artifacts import json_bytes, read_json, training_frame_sha256
from src.cross_validate import CONFIGURATIONS, FOLDS_PATH, N_FOLDS, RESULTS_DIR, load_oof
from src.data_contract import PROJECT_ROOT, SPLIT_SEED, DataValidationError, write_new_files
from src.metrics import probability_metrics
from src.tabpfn_study import baseline_scores, sha256_of, threshold_summary
from src.tune_classical import candidates
from src.tuning_data import CACHE, alternatives, read_score, training_inputs


FAMILIES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}


def cached_vector(name, membership, context, directory=CACHE):
    scores = np.empty(len(membership), dtype=float)
    for fold in range(N_FOLDS):
        valid = membership == fold
        block = read_score(name, fold, int(valid.sum()), context, directory)
        if block is None:
            raise DataValidationError(f"Missing {name} scores for fold {fold + 1}.")
        scores[valid] = block
    return scores


def evaluate(name, family, y, probabilities, **detail):
    return {
        "name": name, "family": family, **detail,
        **probability_metrics(y, probabilities), "thresholds": threshold_summary(y, probabilities),
    }


def by_precision_at_80(entry):
    point = entry["thresholds"]["highest_precision_with_recall_80"]
    return point["precision"], point["recall"], entry["average_precision"]


def by_recall_at_60(entry):
    point = entry["thresholds"]["highest_recall_with_precision_60"]
    return (point["recall"], point["precision"]) if point else (0.0, 0.0)


def by_precision_above_80(entry):
    point = entry["thresholds"]["highest_precision_with_recall_above_80"]
    return (point["precision"], point["recall"], entry["average_precision"]) if point else (0.0, 0.0, 0.0)


def blends(rows, scores, target):
    selected = [
        max((row for row in rows if row["family"] == family), key=by_precision_at_80)["name"]
        for family in sorted({entry["family"] for entry in rows if entry["family"] in (*FAMILIES.values(), "TabPFN")})
    ]
    data = np.column_stack([scores[name] for name in selected])
    log_odds = logit(np.clip(data, 1e-7, 1 - 1e-7))
    rng = np.random.default_rng(SPLIT_SEED)
    weights = [np.full(len(selected), 1 / len(selected)), *rng.dirichlet(np.ones(len(selected)), size=120)]
    for left, right in combinations(range(len(selected)), 2):
        for left_weight in (0.25, 0.5, 0.75):
            weight = np.zeros(len(selected))
            weight[left], weight[right] = left_weight, 1 - left_weight
            weights.append(weight)
    options = []
    for index, weight in enumerate(weights):
        constituent = {name: float(value) for name, value in zip(selected, weight, strict=True) if value > 0}
        for method, blend in (("probability", data @ weight), ("log_odds", expit(log_odds @ weight))):
            options.append(evaluate(
                f"blend_{method}_{index:03}", "Blend", target, blend, method=method, constituent_weights=constituent,
            ))
    for left, right in combinations(range(len(selected)), 2):
        a, b = data[:, left], data[:, right]
        for operation, scores in (
            ("agreement", np.minimum(a, b)),
            ("either", np.maximum(a, b)),
            ("geometric", np.sqrt(a * b)),
        ):
            options.append(evaluate(
                f"blend_{operation}_{left}_{right}", "Blend", target, scores,
                method=operation, constituents=[selected[left], selected[right]],
            ))
    for operation, scores in (
        ("median", np.median(data, axis=1)),
        ("mean_minus_spread", np.clip(data.mean(axis=1) - 0.5 * data.std(axis=1), 0, 1)),
    ):
        options.append(evaluate(f"blend_{operation}", "Blend", target, scores, method=operation, constituents=selected))
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tabpfn", nargs="+", choices=("raw3", "derived1"), required=True)
    parser.add_argument("--extra", action="store_true", help="Include the constrained-risk and stacked development experiments.")
    parser.add_argument("--score-cache", type=Path, default=CACHE)
    parser.add_argument("--tabpfn-reference", type=Path, default=RESULTS_DIR / "tabpfn-3.5-study.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "joint-target-tuning.json")
    args = parser.parse_args()
    train, folds, x, target, membership, context = training_inputs()
    options = [(item.name, item.family) for item in candidates(x, alternatives(x))]
    options += [(f"tabpfn_{configuration}", "TabPFN") for configuration in dict.fromkeys(args.tabpfn)]
    if args.extra:
        from src.tune_operating_point import EXTRA
        options += [(key, family) for key, family, _ in EXTRA]
    tuned_candidates = len(options)
    scores = {name: cached_vector(name, membership, context, args.score_cache) for name, _ in options}
    for original in CONFIGURATIONS:
        name = "original_" + original.key
        options.append((name, FAMILIES[original.model_name]))
        scores[name] = load_oof(train, folds, original, RESULTS_DIR)["p_churn"].to_numpy()
    scores["original_calibrated_xgboost"], _, _ = baseline_scores(train, folds)
    options.append(("original_calibrated_xgboost", "XGBoost"))
    rows = [evaluate(name, family, target, scores[name]) for name, family in options]
    mixture = blends(rows, scores, target)
    previous = read_json(args.tabpfn_reference)
    if (
        previous["folds_sha256"] != sha256_of(FOLDS_PATH)
        or previous["train_frame_sha256"] != training_frame_sha256(train)
        or previous["training_rows"] != len(target)
    ):
        raise DataValidationError("The previous TabPFN report used a different training release or fold assignment.")
    reference = previous["tabpfn_3_5"]
    reference_thresholds = reference["thresholds"].copy()
    reference_thresholds.setdefault(
        "highest_precision_with_recall_above_80",
        reference_thresholds["highest_precision_with_recall_80"]
        if reference_thresholds["highest_precision_with_recall_80"]["recall"] > 0.8 else None,
    )
    rows.append({
        "name": "original_tabpfn_raw1", "family": "TabPFN",
        **{metric: reference[metric] for metric in ("average_precision", "roc_auc", "brier")},
        "thresholds": reference_thresholds,
    })
    all_scores = rows + mixture
    report = {
        "evidence": "Exploratory selection on the same five verified IBM Telco training folds. No held-out test labels were accessed. Reported maxima are exploratory.",
        "customers": len(target), "positives": int(target.sum()), "folds": N_FOLDS, "training_and_folds_sha256": context,
        "precision_target": 0.60, "recall_target": 0.80,
        "candidates": len(rows), "new_candidates": tuned_candidates, "blends": len(mixture),
        "best_by_family": {
            family: max((row for row in all_scores if row["family"] == family), key=by_precision_at_80)
            for family in sorted({row["family"] for row in all_scores})
        },
        "best_precision_at_recall_80": max(all_scores, key=by_precision_at_80),
        "best_precision_at_recall_above_80": max(all_scores, key=by_precision_above_80),
        "best_recall_at_precision_60": max(all_scores, key=by_recall_at_60),
        "target_matches": [row for row in all_scores if row["thresholds"]["meets_both"]],
        "configurations": all_scores,
    }
    write_new_files({args.output: json_bytes(report)})
    print(f"Saved {len(all_scores)} out-of-fold comparisons to {args.output}", flush=True)


if __name__ == "__main__":
    main()
