"""Freeze stratified training folds and score the fixed candidate configurations."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.artifacts import frame_csv, json_bytes, read_json, training_frame_sha256
from src.data_contract import (
    DATA_DIR, FEATURE_COLUMNS, ID_COLUMN, PROJECT_ROOT, SPLIT_SEED, TARGET_COLUMN,
    DataValidationError, parse_csv, write_new_files,
)
from src.preprocessing import MODEL_NAMES, WEIGHTINGS, build_pipeline, churn_probabilities, fitted_settings
from src.train_data import load_train


N_FOLDS = 5
FOLDS_PATH = DATA_DIR / "folds.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FOLD_COLUMNS = (ID_COLUMN, "fold")
OOF_COLUMNS = (*FOLD_COLUMNS, "y_true", "p_churn")


@dataclass(frozen=True)
class ModelConfiguration:
    model_name: str
    weighting: str
    seed: int = SPLIT_SEED

    @property
    def key(self) -> str:
        return f"{self.model_name}_{self.weighting}"

    def as_dict(self) -> dict:
        return asdict(self)

    def build(self):
        return build_pipeline(self.model_name, weighting=self.weighting, seed=self.seed)


CONFIGURATIONS = tuple(ModelConfiguration(model, weight) for model in MODEL_NAMES for weight in WEIGHTINGS)
CONFIG_BY_KEY = {config.key: config for config in CONFIGURATIONS}


def make_folds(train: pd.DataFrame) -> pd.DataFrame:
    ordered = train.sort_values(ID_COLUMN).reset_index(drop=True)
    if ordered[ID_COLUMN].isna().any() or not ordered[ID_COLUMN].is_unique:
        raise DataValidationError("Training fold generation requires unique nonempty row IDs.")
    target = ordered[TARGET_COLUMN]
    classes = target.value_counts()
    if set(classes.index) != {0, 1} or (classes < N_FOLDS).any():
        raise DataValidationError("Five training folds require at least five examples of each binary class.")
    assignments = np.empty(len(ordered), dtype=np.int64)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    for fold, (_, validation) in enumerate(splitter.split(ordered, target)):
        assignments[validation] = fold
    return pd.DataFrame({ID_COLUMN: ordered[ID_COLUMN], "fold": assignments})


def write_folds(train: pd.DataFrame, path: Path = FOLDS_PATH) -> pd.DataFrame:
    folds = make_folds(train)
    write_new_files({path: frame_csv(folds)})
    return folds


def load_folds(train: pd.DataFrame, path: Path = FOLDS_PATH) -> pd.DataFrame:
    rows = parse_csv(path.read_bytes(), FOLD_COLUMNS, "Fold assignments")
    expected = make_folds(train)
    expected_rows = [{ID_COLUMN: row_id, "fold": str(fold)} for row_id, fold in expected.itertuples(index=False, name=None)]
    if sorted(rows, key=lambda row: row[ID_COLUMN]) != expected_rows:
        raise DataValidationError("Fold membership differs from the five pinned stratified training folds.")
    return expected


def training_with_folds(train: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ordered = train.sort_values(ID_COLUMN).reset_index(drop=True)
    expected = make_folds(ordered)
    if not folds.sort_values(ID_COLUMN).reset_index(drop=True).equals(expected):
        raise DataValidationError("Fold membership differs from the five pinned stratified training folds.")
    return ordered, ordered[TARGET_COLUMN].to_numpy(), expected["fold"].to_numpy()


def score_configuration(
    train: pd.DataFrame, folds: pd.DataFrame, config: ModelConfiguration, results_dir: Path = RESULTS_DIR,
) -> pd.DataFrame:
    if config not in CONFIGURATIONS:
        raise DataValidationError("Model configuration is outside the fixed comparison.")
    ordered, labels, membership = training_with_folds(train, folds)
    features = ordered.loc[:, list(FEATURE_COLUMNS)]
    probabilities = np.empty(len(ordered), dtype=float)
    fold_settings = []
    for fold in range(N_FOLDS):
        validation = membership == fold
        pipeline = config.build().fit(features.loc[~validation], labels[~validation])
        probabilities[validation] = churn_probabilities(pipeline, features.loc[validation])
        fold_settings.append({"fold": fold, "settings": fitted_settings(pipeline)})

    oof = pd.DataFrame({ID_COLUMN: ordered[ID_COLUMN], "fold": membership, "y_true": labels, "p_churn": probabilities})
    payload = frame_csv(oof)
    metadata = {
        "configuration": config.as_dict(), "train_frame_sha256": training_frame_sha256(train),
        "predictions_sha256": sha256(payload).hexdigest(), "fold_settings": fold_settings,
    }
    output = results_dir / "oof"
    write_new_files({output / f"{config.key}.csv": payload, output / f"{config.key}_settings.json": json_bytes(metadata)})
    return oof


def load_oof(train: pd.DataFrame, folds: pd.DataFrame, config: ModelConfiguration, results_dir: Path) -> pd.DataFrame:
    ordered, labels, membership = training_with_folds(train, folds)
    output = results_dir / "oof"
    metadata = read_json(output / f"{config.key}_settings.json")
    payload = (output / f"{config.key}.csv").read_bytes()
    if (metadata.get("configuration") != config.as_dict()
        or metadata.get("train_frame_sha256") != training_frame_sha256(train)
        or metadata.get("predictions_sha256") != sha256(payload).hexdigest()):
        raise DataValidationError(f"OOF provenance differs for {config.key}.")
    scores = pd.read_csv(BytesIO(payload), dtype={ID_COLUMN: str, "p_churn": float}, float_precision="round_trip")
    if tuple(scores.columns) != OOF_COLUMNS:
        raise DataValidationError(f"OOF header differs for {config.key}.")
    scores = scores.sort_values(ID_COLUMN)
    if (not np.array_equal(scores[ID_COLUMN], ordered[ID_COLUMN])
        or not np.array_equal(scores["fold"], membership) or not np.array_equal(scores["y_true"], labels)
        or not scores["p_churn"].between(0, 1).all()):
        raise DataValidationError(f"OOF rows or probabilities differ from the verified training folds for {config.key}.")
    return scores.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-folds", action="store_true", help="Create the assignment file and exit")
    parser.add_argument("--folds", type=Path, default=FOLDS_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--configuration", action="append", choices=CONFIG_BY_KEY, help="Score only these named configurations")
    args = parser.parse_args()
    try:
        train = load_train()
        if args.write_folds:
            write_folds(train, args.folds)
            print(f"Wrote five stratified training folds to {args.folds}.")
            return
        folds = load_folds(train, args.folds)
        for config in (CONFIG_BY_KEY[key] for key in args.configuration or CONFIG_BY_KEY):
            score_configuration(train, folds, config, args.results_dir)
            print(f"Scored {config.key} on all five folds.")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cross-validation failed: {error}\n")


if __name__ == "__main__":
    main()
