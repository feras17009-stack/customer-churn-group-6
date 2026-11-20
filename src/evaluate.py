"""Score the private holdout once per freeze and publish aggregate metrics."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import json_bytes
from src.data_contract import (
    DATA_DIR, FEATURE_COLUMNS, IBM_PIN, ID_COLUMN, LABELED_COLUMNS, PROJECT_ROOT, SPLIT_SEED, TARGET_COLUMN,
    DatasetPin, DataValidationError, parse_csv, validate_predictors, write_new_files,
)
from src.freeze_model import load_frozen_model
from src.metrics import operating_metrics, probability_metrics
from src.policy import ContactPolicy, CostScenario, contact_decisions, customer_costs, probability_array
from src.preprocessing import churn_probabilities
from src.train_data import partition_ids


BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_COVERAGE = 0.95


def _cost_summary(values: np.ndarray) -> dict:
    return {"total": float(values.sum()), "per_customer": float(values.mean())}


def _bootstrap(
    labels: np.ndarray, scores: np.ndarray, contacts: np.ndarray, costs: dict[str, np.ndarray], savings: dict[str, np.ndarray],
) -> dict:
    rng = np.random.default_rng(SPLIT_SEED)
    samples: dict[str, list[float]] = {}
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, labels.size, size=labels.size)
        y = labels[indices]
        operating = operating_metrics(y, contacts[indices])
        measurements = {
            **probability_metrics(y, scores[indices]),
            **{metric: operating[metric] for metric in ("precision", "recall", "f1")},
            **{f"{policy}_per_customer_cost": float(values[indices].mean()) for policy, values in costs.items()},
            **{f"per_customer_saving_{baseline}": float(values[indices].mean()) for baseline, values in savings.items()},
        }
        for metric, value in measurements.items():
            values = samples.setdefault(metric, [])
            if value is not None:
                values.append(value)
    intervals = {}
    for metric, values in samples.items():
        tails = (1 - BOOTSTRAP_COVERAGE) / 2
        lower, upper = np.quantile(values, [tails, 1 - tails]).tolist() if values else (None, None)
        intervals[metric] = {"lower": lower, "upper": upper, "valid_replicates": len(values)}
    return {
        "method": "paired customer-level percentile", "resamples": BOOTSTRAP_REPLICATES, "seed": SPLIT_SEED,
        "coverage": BOOTSTRAP_COVERAGE, "intervals": intervals,
    }


def evaluate_scores(labels, probabilities, policy: ContactPolicy, scenario: CostScenario) -> dict:
    y = np.asarray(labels)
    p = probability_array(probabilities)
    if y.shape != p.shape or y.size == 0 or not np.isin(y, [0, 1]).all():
        raise DataValidationError("Evaluation requires matching nonempty probabilities and binary labels.")
    contacts = contact_decisions(p, policy)
    operating = operating_metrics(y, contacts)
    costs = {
        "frozen_policy": customer_costs(y, contacts, scenario),
        "contact_all": customer_costs(y, np.ones(y.shape, dtype=bool), scenario),
        "contact_none": customer_costs(y, np.zeros(y.shape, dtype=bool), scenario),
    }
    savings = {f"vs_{baseline}": costs[baseline] - costs["frozen_policy"] for baseline in ("contact_all", "contact_none")}
    precision, recall = operating["precision"], operating["recall"]
    precision_reached = precision >= 0.6 if precision is not None else None
    recall_reached = recall >= 0.8 if recall is not None else None
    return {
        **operating, **probability_metrics(y, p), "policy": policy.as_dict(), "cost_scenario": scenario.as_dict(),
        "precision_status": "undefined (zero contacts)" if precision is None else "defined",
        "scenario_costs": {key: _cost_summary(values) for key, values in costs.items()},
        "paired_savings": {key: _cost_summary(values) for key, values in savings.items()},
        "bootstrap": _bootstrap(y, p, contacts, costs, savings),
        "exploratory_targets": {
            "minimum_precision": 0.6, "minimum_recall": 0.8, "reaches_precision": precision_reached,
            "reaches_recall": recall_reached, "reaches_both": precision_reached is True and recall_reached is True,
            "uncertainty": "See the paired bootstrap precision and recall intervals.",
        },
        "interpretation": "Independent performance on the frozen IBM Telco snapshot holdout. Costs use the stated scenario assumptions.",
    }


def _parse_holdout(payload: bytes, expected_ids: set[str]) -> pd.DataFrame:
    rows = parse_csv(payload, LABELED_COLUMNS, "Private holdout")
    ids = set()
    for index, row in enumerate(rows, start=2):
        customer_id = row[ID_COLUMN]
        if customer_id not in expected_ids or customer_id in ids:
            raise DataValidationError(f"Private holdout has an unexpected or duplicate row_id at line {index}.")
        ids.add(customer_id)
        validate_predictors(row, f"Private holdout line {index}")
        if row[TARGET_COLUMN] not in ("0", "1"):
            raise DataValidationError(f"Private holdout has an invalid binary target at line {index}.")
    if ids != expected_ids:
        raise DataValidationError("Private holdout rows do not cover the exact frozen holdout IDs.")
    return pd.DataFrame(rows, columns=LABELED_COLUMNS).sort_values(ID_COLUMN).reset_index(drop=True)


def evaluate_holdout(
    holdout: Path, manifest: Path, output: Path, *, splits: Path = DATA_DIR / "splits.csv", pin: DatasetPin = IBM_PIN,
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Evaluation output already exists: {output}.")
    holdout = holdout.resolve()
    if holdout.is_relative_to(PROJECT_ROOT):
        raise DataValidationError("The private holdout must stay outside the project directory.")
    frozen = load_frozen_model(manifest)
    split_bytes = splits.read_bytes()
    if sha256(split_bytes).hexdigest() != frozen.decision.get("split_manifest_sha256"):
        raise DataValidationError("Split manifest SHA-256 does not match the freeze.")
    _, expected_ids = partition_ids(split_bytes, pin)
    if not holdout.is_file():
        raise FileNotFoundError(f"Private holdout file is unavailable: {holdout}.")

    freeze_id = frozen.decision["freeze_id"]
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = holdout.parent / ".churn-evaluations" / f"{freeze_id}.json"
    if audit.exists() or audit.is_symlink():
        raise FileExistsError(f"An evaluation has already started for {freeze_id} in this custodian environment.")
    with holdout.open("rb") as stream:
        write_new_files({audit: json_bytes({"freeze_id": freeze_id, "started_utc": datetime.now(timezone.utc).isoformat()})})
        frame = _parse_holdout(stream.read(), expected_ids)
    scores = churn_probabilities(frozen.estimator, frame.loc[:, list(FEATURE_COLUMNS)])
    result = {
        "freeze_id": freeze_id, "configuration": frozen.decision["configuration"],
        **evaluate_scores(frame[TARGET_COLUMN].astype(int), scores, frozen.policy, frozen.scenario),
    }
    write_new_files({output: json_bytes(result)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", required=True, type=Path, help="Restricted labeled holdout produced by intake")
    parser.add_argument("--manifest", required=True, type=Path, help="Trusted freeze manifest outside the holdout directory")
    parser.add_argument("--output", required=True, type=Path, help="New aggregate JSON output")
    parser.add_argument("--splits", type=Path, default=DATA_DIR / "splits.csv", help="Committed split manifest for this freeze")
    args = parser.parse_args()
    try:
        result = evaluate_holdout(args.holdout, args.manifest, args.output, splits=args.splits, pin=IBM_PIN)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Holdout evaluation failed: {error}\n")
    print(f"Published aggregate metrics for {result['freeze_id']} to {args.output}.")


if __name__ == "__main__":
    main()
