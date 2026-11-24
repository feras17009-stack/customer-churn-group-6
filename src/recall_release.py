"""Reproduce and freeze the development winner with recall strictly above 80%."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from src.artifacts import read_json, training_frame_sha256
from src.cross_validate import N_FOLDS, RESULTS_DIR
from src.data_contract import DATA_DIR, DataValidationError
from src.freeze_model import APP_MANIFEST_PATH, CLASS_MAPPING, load_frozen_model, runtime_versions, write_estimator_bundle
from src.policy import ContactPolicy, load_cost_scenario
from src.preprocessing import churn_probabilities
from src.recall_blend import BLEND_CONFIGURATION, BLEND_MEMBERS, BLEND_NAME, RecallBlend
from src.tabpfn_study import RECALL_TARGET, sha256_of, threshold_summary
from src.tune_classical import candidates
from src.tuning_data import alternatives, training_inputs


def release_estimators(features):
    configured = {candidate.name: candidate for candidate in candidates(features, alternatives(features))}
    return tuple(configured[name].build() for name in BLEND_MEMBERS)


def recreate_threshold(features, labels, membership, estimators):
    probabilities = np.empty(len(labels), dtype=float)
    for fold in range(N_FOLDS):
        validation = membership == fold
        model = RecallBlend(estimators).fit(features.loc[~validation], labels[~validation])
        probabilities[validation] = churn_probabilities(model, features.loc[validation])
        print(f"Verified the selected blend on fold {fold + 1}/{N_FOLDS}.", flush=True)
    return threshold_summary(labels, probabilities)["highest_precision_with_recall_above_80"]


def freeze_recall_release(report_path: Path, output_dir: Path) -> Path:
    train, _, features, labels, membership, context = training_inputs()
    report = read_json(report_path)
    winner = report.get("best_precision_at_recall_above_80", {})
    if (
        report.get("training_and_folds_sha256") != context
        or winner.get("name") != BLEND_NAME
        or winner.get("method") != BLEND_CONFIGURATION["method"]
        or winner.get("constituent_weights") != BLEND_CONFIGURATION["members"]
    ):
        raise DataValidationError("The development report does not match the approved blend and training folds.")
    published_point = winner["thresholds"]["highest_precision_with_recall_above_80"]
    estimators = release_estimators(features)
    point = recreate_threshold(features, labels, membership, estimators)
    if (
        point is None
        or point["recall"] <= RECALL_TARGET
        or point["confusion_matrix"] != published_point["confusion_matrix"]
        or not np.isclose(point["threshold"], published_point["threshold"], rtol=1e-7, atol=1e-8)
    ):
        raise DataValidationError("The refit did not reproduce the selected development operating point.")
    frozen = RecallBlend(estimators).fit(features, labels)
    decision = {
        "freeze_id": "freeze-v2",
        "class_mapping": CLASS_MAPPING,
        "positive_class": 1,
        "configuration": BLEND_CONFIGURATION,
        "calibration_recipe": None,
        "policy": ContactPolicy("threshold", point["threshold"]).as_dict(),
        "cost_scenario": load_cost_scenario().as_dict(),
        "runtime_versions": runtime_versions(),
        "train_frame_sha256": training_frame_sha256(train),
        "split_manifest_sha256": sha256((DATA_DIR / "splits.csv").read_bytes()).hexdigest(),
        "evaluation_scope": "training_oof",
        "development_selection": {
            "report_sha256": sha256_of(report_path),
            "recall_floor_exclusive": RECALL_TARGET,
            "operating_point": point,
        },
    }
    manifest = write_estimator_bundle(output_dir, frozen, decision)
    load_frozen_model(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=RESULTS_DIR / "joint-target-tuning.json")
    parser.add_argument("--model-dir", type=Path, default=APP_MANIFEST_PATH.parent)
    args = parser.parse_args()
    try:
        manifest = freeze_recall_release(args.report, args.model_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Recall release failed: {error}\n")
    print(f"Created and verified freeze-v2 at {manifest}.", flush=True)


if __name__ == "__main__":
    main()
