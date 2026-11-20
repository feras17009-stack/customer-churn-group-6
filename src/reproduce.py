"""Reproduce the training-only run and execute the notebooks in fresh kernels."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

from src.artifacts import json_bytes, read_json, training_frame_sha256
from src.calibrate import build_calibrated, write_calibration
from src.compare_models import compare_models, load_selected_model
from src.cross_validate import CONFIGURATIONS, FOLDS_PATH, RESULTS_DIR, load_folds, load_oof, score_configuration
from src.data_contract import FEATURE_COLUMNS, ID_COLUMN, PROJECT_ROOT, TARGET_COLUMN, DataValidationError, write_new_files
from src.freeze_model import MANIFEST_PATH, load_frozen_model, runtime_versions
from src.policy import ContactPolicy, contact_decisions, load_cost_scenario
from src.preprocessing import churn_probabilities
from src.train_data import load_train


def _score_difference(expected, reproduced, name: str) -> float:
    if not np.allclose(expected, reproduced, rtol=1e-7, atol=1e-8):
        raise DataValidationError(f"Reproduced probabilities exceed the comparison tolerance for {name}.")
    return float(np.max(np.abs(np.asarray(expected) - np.asarray(reproduced))))


def verify_training_run(
    train: pd.DataFrame, folds: pd.DataFrame, results_dir: Path, manifest: Path, scratch_dir: Path,
) -> dict:
    frozen = load_frozen_model(manifest)
    selected = load_selected_model(train, results_dir / "selected_model.json")
    if frozen.decision["train_frame_sha256"] != training_frame_sha256(train) or frozen.decision["configuration"] != selected.as_dict():
        raise DataValidationError("Frozen model differs from the verified training frame or selected configuration.")
    maximum_differences = {}
    for config in CONFIGURATIONS:
        expected = load_oof(train, folds, config, results_dir)
        new_oof = score_configuration(train, folds, config, scratch_dir)
        maximum_differences[config.key] = _score_difference(expected["p_churn"], new_oof["p_churn"], config.key)
    _, repeated_selection = compare_models(train, folds, scratch_dir)
    if repeated_selection["configuration"] != selected.as_dict():
        raise DataValidationError("The independent run selected a different configuration.")

    original_dir = results_dir / "calibration"
    original = read_json(original_dir / "report.json")
    scenario = load_cost_scenario()
    repeated = write_calibration(train, folds, selected, scenario, scratch_dir / "calibration")
    old_oof = pd.read_csv(original_dir / "oof.csv", dtype={ID_COLUMN: str}, float_precision="round_trip")
    new_oof = pd.read_csv(scratch_dir / "calibration" / "oof.csv", dtype={ID_COLUMN: str}, float_precision="round_trip")
    if not old_oof[[ID_COLUMN, "fold", "y_true"]].equals(new_oof[[ID_COLUMN, "fold", "y_true"]]):
        raise DataValidationError("Reproduced calibration rows do not match the original training blocks.")
    maximum_differences["calibrated_oof"] = _score_difference(old_oof["p_churn"], new_oof["p_churn"], "calibrated OOF scores")
    primary_policy = ContactPolicy.from_dict(original["policy_selection"]["policy"])
    reproduced_policy = ContactPolicy.from_dict(repeated["policy_selection"]["policy"])
    if (primary_policy != frozen.policy or scenario != frozen.scenario
        or original["policy_selection"]["method"] != repeated["policy_selection"]["method"]
        or primary_policy.policy_type != reproduced_policy.policy_type
        or not np.array_equal(contact_decisions(old_oof["p_churn"], primary_policy), contact_decisions(new_oof["p_churn"], reproduced_policy))):
        raise DataValidationError("The independent run did not reproduce the frozen outreach policy and decisions.")

    features = train.loc[:, list(FEATURE_COLUMNS)]
    refit = build_calibrated(selected).fit(features, train[TARGET_COLUMN])
    maximum_differences["final_model_on_training"] = _score_difference(
        churn_probabilities(frozen.estimator, features), churn_probabilities(refit, features), "reloaded final model",
    )
    return {"configuration": selected.as_dict(), "max_absolute_probability_differences": maximum_differences, "runtime_versions": runtime_versions()}


def execute_notebook(path: Path, *, working_dir: Path = PROJECT_ROOT):
    notebook = nbformat.read(path, as_version=4)
    return NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(working_dir)}}).execute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / ".scratch" / "reproduction")
    args = parser.parse_args()
    try:
        outputs = {}
        if not args.notebooks_only:
            train = load_train()
            with TemporaryDirectory(prefix="churn-reproduce-") as temporary:
                receipt = verify_training_run(train, load_folds(train, FOLDS_PATH), RESULTS_DIR, MANIFEST_PATH, Path(temporary))
            outputs[args.output_dir / "training_reproduction.json"] = json_bytes(receipt)
        for filename in ("eda.ipynb", "tutorial.ipynb"):
            completed = execute_notebook(PROJECT_ROOT / "notebooks" / filename)
            outputs[args.output_dir / filename] = nbformat.writes(completed).encode("utf-8")
        write_new_files(outputs)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Reproduction failed: {error}\n")
    print(f"Saved independent verification to {args.output_dir}.")


if __name__ == "__main__":
    main()
