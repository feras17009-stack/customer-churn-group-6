"""Refit on the verified training release and verify the frozen model bundle."""

import argparse
import platform
import re
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.artifacts import json_bytes, parse_json_object, read_json, training_frame_sha256
from src.calibrate import CALIBRATION_RECIPE, build_calibrated
from src.compare_models import load_selected_model
from src.cross_validate import CONFIGURATIONS, RESULTS_DIR, ModelConfiguration
from src.data_contract import DATA_DIR, FEATURE_COLUMNS, PROJECT_ROOT, TARGET_COLUMN, DataValidationError, write_new_files
from src.input_schema import SCHEMA_PATH, parse_input_schema
from src.policy import COST_SCENARIO_PATH, ContactPolicy, CostScenario, load_cost_scenario
from src.preprocessing import fitted_settings
from src.train_data import load_train


MODEL_DIR = PROJECT_ROOT / "models"
MANIFEST_NAME = "freeze_manifest.json"
MANIFEST_PATH = MODEL_DIR / MANIFEST_NAME
MODEL_NAME = "churn_model.joblib"
DECISION_NAME = "decision.json"
SCHEMA_NAME = "input_schema.json"
CLASS_MAPPING = {"0": "No", "1": "Yes"}
BUNDLE_FILES = (MODEL_NAME, DECISION_NAME, SCHEMA_NAME)


@dataclass(frozen=True)
class FrozenModel:
    estimator: CalibratedClassifierCV
    schema: dict
    decision: dict
    policy: ContactPolicy
    scenario: CostScenario


def runtime_versions() -> dict[str, str]:
    try:
        xgboost = version("xgboost-cpu")
    except PackageNotFoundError:
        xgboost = version("xgboost")
    names = ("catboost", "joblib", "numpy", "pandas", "scikit-learn", "scipy", "streamlit", "threadpoolctl")
    return {"python": platform.python_version(), **{name: version(name) for name in names}, "xgboost": xgboost}


def _validate_freeze_id(freeze_id) -> None:
    if not isinstance(freeze_id, str) or not re.fullmatch(r"freeze-v[1-9]\d*", freeze_id):
        raise DataValidationError("Freeze ID must have the form freeze-v1, freeze-v2, and so on.")


def write_model_bundle(
    train: pd.DataFrame, config: ModelConfiguration, scenario: CostScenario, policy: ContactPolicy,
    model_dir: Path, splits_path: Path, *, freeze_id: str = "freeze-v1",
) -> Path:
    _validate_freeze_id(freeze_id)
    if config not in CONFIGURATIONS:
        raise DataValidationError("Freeze requires one of the eight fixed model configurations.")
    schema_path = model_dir / SCHEMA_NAME
    existing_schema = schema_path.is_file()
    schema_payload = (schema_path if existing_schema else SCHEMA_PATH).read_bytes()
    parse_input_schema(schema_payload)
    split_digest = sha256(splits_path.read_bytes()).hexdigest()

    features = train.loc[:, list(FEATURE_COLUMNS)]
    estimator = build_calibrated(config).fit(features, train[TARGET_COLUMN])
    calibrated_pipeline = estimator.calibrated_classifiers_[0].estimator
    decision = {
        "freeze_id": freeze_id, "class_mapping": CLASS_MAPPING, "positive_class": 1,
        "configuration": config.as_dict(), "resolved_estimator_settings": fitted_settings(calibrated_pipeline),
        "calibration_recipe": CALIBRATION_RECIPE, "policy": policy.as_dict(), "cost_scenario": scenario.as_dict(),
        "runtime_versions": runtime_versions(), "train_frame_sha256": training_frame_sha256(train),
        "split_manifest_sha256": split_digest,
    }
    model_file = BytesIO()
    joblib.dump(estimator, model_file, compress=3)
    payloads = {MODEL_NAME: model_file.getvalue(), DECISION_NAME: json_bytes(decision), SCHEMA_NAME: schema_payload}
    manifest = {"freeze_id": freeze_id, "files": {name: sha256(content).hexdigest() for name, content in payloads.items()}}
    outputs = {model_dir / name: payload for name, payload in payloads.items() if name != SCHEMA_NAME or not existing_schema}
    manifest_path = model_dir / MANIFEST_NAME
    outputs[manifest_path] = json_bytes(manifest)
    write_new_files(outputs)
    return manifest_path


def verify_freeze(manifest_path: Path = MANIFEST_PATH) -> tuple[dict, dict[str, bytes]]:
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    _validate_freeze_id(manifest.get("freeze_id"))
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(BUNDLE_FILES):
        raise DataValidationError("Freeze manifest must enumerate the model, decision and input schema.")
    payloads = {}
    for name in BUNDLE_FILES:
        content = (manifest_path.parent / name).read_bytes()
        if sha256(content).hexdigest() != files[name]:
            raise DataValidationError(f"Freeze SHA-256 mismatch in {name}.")
        payloads[name] = content
    return manifest, payloads


def load_frozen_model(manifest_path: Path = MANIFEST_PATH) -> FrozenModel:
    """Load only a trusted team bundle, after verifying the exact bytes to be deserialized."""
    manifest, payloads = verify_freeze(manifest_path)
    decision = parse_json_object(payloads[DECISION_NAME], DECISION_NAME)
    if decision.get("freeze_id") != manifest["freeze_id"] or decision.get("class_mapping") != CLASS_MAPPING or decision.get("positive_class") != 1:
        raise DataValidationError("Frozen decision must match its freeze ID and binary label mapping.")
    if decision.get("runtime_versions") != runtime_versions():
        raise DataValidationError("Frozen model requires the Python and library versions in decision.json.")
    if decision.get("configuration") not in [config.as_dict() for config in CONFIGURATIONS] or decision.get("calibration_recipe") != CALIBRATION_RECIPE:
        raise DataValidationError("Frozen model configuration or calibration recipe is invalid.")
    schema = parse_input_schema(payloads[SCHEMA_NAME])
    policy = ContactPolicy.from_dict(decision.get("policy"))
    scenario = CostScenario.from_dict(decision.get("cost_scenario"))
    estimator = joblib.load(BytesIO(payloads[MODEL_NAME]))
    if not isinstance(estimator, CalibratedClassifierCV) or not np.array_equal(estimator.classes_, [0, 1]):
        raise DataValidationError("Frozen model must be a calibrated classifier with classes 0 and 1.")
    return FrozenModel(estimator, schema, decision, policy, scenario)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify and reload an existing trusted model")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--scenario", type=Path, default=COST_SCENARIO_PATH)
    parser.add_argument("--freeze-id", default="freeze-v1")
    args = parser.parse_args()
    try:
        if args.verify:
            frozen = load_frozen_model(args.model_dir / MANIFEST_NAME)
            print(f"Verified and reloaded {frozen.decision['freeze_id']}.")
            return
        train = load_train()
        config = load_selected_model(train, args.results_dir / "selected_model.json")
        scenario = load_cost_scenario(args.scenario)
        report = read_json(args.results_dir / "calibration" / "report.json")
        if (report.get("configuration") != config.as_dict() or report.get("primary_scenario") != scenario.as_dict()
            or report.get("train_frame_sha256") != training_frame_sha256(train) or report.get("recipe") != CALIBRATION_RECIPE):
            raise DataValidationError("Calibration report does not match this training data, model, recipe and cost scenario.")
        policy = ContactPolicy.from_dict(report.get("policy_selection", {}).get("policy"))
        path = write_model_bundle(train, config, scenario, policy, args.model_dir, DATA_DIR / "splits.csv", freeze_id=args.freeze_id)
        load_frozen_model(path)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Model freeze failed: {error}\n")
    print(f"Created and verified {args.freeze_id} at {path}.")


if __name__ == "__main__":
    main()
