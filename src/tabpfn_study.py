"""Compare TabPFN-3.5 and the frozen model recipe on verified IBM training folds."""

import argparse
from hashlib import file_digest
from importlib.metadata import version
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.artifacts import json_bytes, parse_json_object, read_json, training_frame_sha256
from src.calibrate import CALIBRATION_RECIPE
from src.cross_validate import FOLDS_PATH, N_FOLDS, RESULTS_DIR, ModelConfiguration, load_folds, load_oof, training_with_folds
from src.data_contract import FEATURE_COLUMNS, ID_COLUMN, PROJECT_ROOT, SPLIT_SEED, DataValidationError, write_new_files
from src.freeze_model import DECISION_NAME, verify_freeze
from src.metrics import operating_metrics, probability_metrics
from src.policy import ContactPolicy, contact_decisions, probability_array
from src.preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, churn_probabilities
from src.train_data import load_train


CHECKPOINT = "tabpfn-v3.5-20260909.safetensors"
CHECKPOINT_SHA256 = "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"
PRECISION_TARGET = 0.60
RECALL_TARGET = 0.80
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "tabpfn-3.5-study.json"


def sha256_of(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def threshold_summary(labels, probabilities) -> dict:
    scores = probability_array(probabilities)
    target = np.asarray(labels)
    if target.shape != scores.shape or set(target) != {0, 1}:
        raise DataValidationError("Threshold analysis requires matching scores and both binary classes.")
    precision, recall, thresholds = precision_recall_curve(target, scores)
    precision, recall = precision[:-1], recall[:-1]
    eligible_precision = precision >= PRECISION_TARGET
    eligible_recall = recall >= RECALL_TARGET
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=precision + recall > 0)

    def best(mask, priority):
        points = np.flatnonzero(mask)
        if points.size == 0:
            return None
        index = max(points, key=priority)
        cutoff = float(thresholds[index])
        contact = contact_decisions(scores, ContactPolicy("threshold", cutoff))
        return {"threshold": cutoff, **operating_metrics(target, contact)}

    return {
        "meets_both": best(eligible_precision & eligible_recall, lambda index: (f1[index], precision[index], thresholds[index])),
        "highest_precision_with_recall_80": best(eligible_recall, lambda index: (precision[index], recall[index], thresholds[index])),
        "highest_precision_with_recall_above_80": best(recall > RECALL_TARGET, lambda index: (precision[index], recall[index], thresholds[index])),
        "highest_recall_with_precision_60": best(eligible_precision, lambda index: (recall[index], precision[index], thresholds[index])),
        "max_f1": best(np.ones(len(thresholds), dtype=bool), lambda index: (f1[index], precision[index], thresholds[index])),
    }


def summaries(labels, membership, scores, policy: ContactPolicy) -> dict:
    decisions = contact_decisions(scores, policy)
    return {
        **probability_metrics(labels, scores),
        "fold_metrics": [
            {"fold": fold, **probability_metrics(labels[membership == fold], scores[membership == fold])}
            for fold in range(N_FOLDS)
        ],
        "reference_operating_point": {"policy": policy.as_dict(), **operating_metrics(labels, decisions)},
        "thresholds": threshold_summary(labels, scores),
    }


def baseline_scores(train, folds) -> tuple[np.ndarray, ContactPolicy, dict]:
    refit, files = verify_freeze()
    decision = parse_json_object(files[DECISION_NAME], DECISION_NAME)
    provenance = read_json(RESULTS_DIR / "calibration" / "report.json")
    digest = training_frame_sha256(train)
    if (
        decision["configuration"] != provenance["configuration"]
        or decision["train_frame_sha256"] != digest
        or provenance["train_frame_sha256"] != digest
        or decision["calibration_recipe"] != CALIBRATION_RECIPE
        or provenance["recipe"] != CALIBRATION_RECIPE
        or decision["policy"] != provenance["policy_selection"]["policy"]
    ):
        raise DataValidationError("Calibration metadata does not match the verified training release and frozen decision.")
    config = ModelConfiguration(**decision["configuration"])
    baseline = load_oof(train, folds, config, RESULTS_DIR)
    score_path = RESULTS_DIR / "calibration" / "oof.csv"
    calibrated = pd.read_csv(score_path, dtype={ID_COLUMN: str}, float_precision="round_trip")
    calibrated = calibrated.sort_values(ID_COLUMN).reset_index(drop=True)
    keys = [ID_COLUMN, "fold", "y_true"]
    if not calibrated[keys].equals(baseline[keys]) or not np.allclose(
        calibrated["p_baseline"], baseline["p_churn"], rtol=1e-7, atol=1e-8,
    ):
        raise DataValidationError("Calibrated scores do not align with the verified uncalibrated scores and fold members.")
    return probability_array(calibrated["p_churn"]), ContactPolicy.from_dict(decision["policy"]), {
        "freeze_id": refit["freeze_id"], "configuration": decision["configuration"], "score_file_sha256": sha256_of(score_path),
    }


def tabpfn_scores(table, labels, membership, checkpoint: Path, device: str, estimators: int | None):
    os.environ["TABPFN_NO_BROWSER"] = "1"
    try:
        import torch
        from tabpfn import TabPFNClassifier
    except ImportError as error:
        raise DataValidationError("Install the optional TabPFN package and PyTorch in an experiment environment.") from error
    if device == "cuda" and not torch.cuda.is_available():
        raise DataValidationError("This installation does not expose a CUDA GPU. Use --device cpu or a CUDA-enabled PyTorch installation.")
    if device == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))

    features = table.loc[:, list(FEATURE_COLUMNS)].copy()
    for column in NUMERIC_COLUMNS:
        numeric = features[column].replace(r"^\s*$", np.nan, regex=True)
        features[column] = pd.to_numeric(numeric, errors="raise")
    categories = [features.columns.get_loc(name) for name in CATEGORICAL_COLUMNS]
    parameters = {
        "device": device,
        "model_path": checkpoint,
        "random_state": SPLIT_SEED,
        "categorical_features_indices": categories,
        "memory_saving_mode": "auto",
    }
    if estimators is not None:
        parameters["n_estimators"] = estimators
    probabilities = np.empty(len(labels), dtype=float)
    timings = []
    for fold in range(N_FOLDS):
        validation = membership == fold
        estimator = TabPFNClassifier(**parameters)
        print(f"Scoring fold {fold + 1}/{N_FOLDS} on {device} with {int((~validation).sum())} training rows.", flush=True)
        tick = perf_counter()
        estimator.fit(features.loc[~validation], labels[~validation])
        probabilities[validation] = churn_probabilities(estimator, features.loc[validation])
        timings.append({
            "fold": fold, "train_rows": int((~validation).sum()), "validation_rows": int(validation.sum()),
            "seconds": round(perf_counter() - tick, 2),
        })
        print(f"Scored fold {fold + 1}/{N_FOLDS} in {timings[-1]['seconds']:.1f}s.", flush=True)
    return probabilities, {
        "version": version("tabpfn"), "torch_version": torch.__version__, "model": "TabPFN-3.5",
        "checkpoint": CHECKPOINT, "checkpoint_sha256": CHECKPOINT_SHA256, "device": device,
        "n_estimators": estimator.n_estimators, "random_state": SPLIT_SEED, "timings": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--checkpoint", type=Path, help="Path to the official TabPFN-3.5 checkpoint previously obtained under its license.")
    sources.add_argument("--baseline-only", action="store_true", help="Verify and summarize the current XGBoost training-fold reference.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--estimators", type=int, help="Number of TabPFN ensemble members. The default uses the package setting.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.estimators is not None and args.estimators < 1:
        parser.error("--estimators must be positive.")
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        if args.checkpoint:
            if args.checkpoint.name != CHECKPOINT or sha256_of(args.checkpoint) != CHECKPOINT_SHA256:
                raise DataValidationError(f"Expected the official {CHECKPOINT} with its published SHA-256.")
        train = load_train()
        folds = load_folds(train)
        ordered, labels, membership = training_with_folds(train, folds)
        scores, policy, metadata = baseline_scores(train, folds)
        report = {
            "scope": "Development results from the five verified IBM Telco training folds. Thresholds are explored on pooled out-of-fold scores. An independent sample is required for confirmation.",
            "training_rows": len(train), "folds_sha256": sha256_of(FOLDS_PATH),
            "train_frame_sha256": training_frame_sha256(train),
            "targets": {"precision": PRECISION_TARGET, "recall": RECALL_TARGET},
            "calibrated_xgboost": {**metadata, **summaries(labels, membership, scores, policy)},
        }
        if args.checkpoint:
            challenger, runtime = tabpfn_scores(ordered, labels, membership, args.checkpoint.resolve(), args.device, args.estimators)
            report["tabpfn_3_5"] = {
                **runtime, **summaries(labels, membership, challenger, ContactPolicy("threshold", 0.5)),
            }
        write_new_files({args.output: json_bytes(report)})
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"TabPFN study failed: {error}\n")
    print(f"Saved the development comparison to {args.output}")


if __name__ == "__main__":
    main()
