"""Training-only inputs and score storage for local churn experiments."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import training_frame_sha256
from src.cross_validate import load_folds, training_with_folds
from src.data_contract import FEATURE_COLUMNS, PROJECT_ROOT, SPLIT_SEED, DataValidationError
from src.preprocessing import NUMERIC_COLUMNS
from src.train_data import load_train


CACHE = PROJECT_ROOT / ".scratch" / "joint-target-tuning"


def prepare_features(table: pd.DataFrame) -> pd.DataFrame:
    features = table.loc[:, list(FEATURE_COLUMNS)].copy()
    for name in NUMERIC_COLUMNS:
        features[name] = pd.to_numeric(features[name].replace(r"^\s*$", np.nan, regex=True), errors="raise")
    features["SeniorCitizen"] = features["SeniorCitizen"].astype(str)
    return features


def training_inputs():
    train = load_train()
    folds = load_folds(train)
    table, y, membership = training_with_folds(train, folds)
    x = prepare_features(table)
    data_digest = training_frame_sha256(train)
    context = sha256(data_digest.encode() + membership.astype("<i8").tobytes()).hexdigest()
    return train, folds, x, y, membership, context


def alternatives(x: pd.DataFrame):
    engineered = x.copy()
    tenure = x["tenure"].astype(float).where(x["tenure"] > 0)
    historical = x["TotalCharges"] / tenure
    engineered["log_tenure"] = np.log1p(x["tenure"])
    engineered["historical_monthly_charge"] = historical
    engineered["price_minus_historical"] = x["MonthlyCharges"] - historical
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    engineered["number_of_internet_addons"] = x[addons].eq("Yes").sum(axis=1)
    engineered["security_and_support_count"] = x[["OnlineSecurity", "TechSupport"]].eq("Yes").sum(axis=1)
    engineered["streaming_count"] = x[["StreamingTV", "StreamingMovies"]].eq("Yes").sum(axis=1)
    engineered["months_since_contract_milestone"] = np.select(
        [x["Contract"].eq("One year"), x["Contract"].eq("Two year")],
        [x["tenure"] % 12, x["tenure"] % 24],
        default=0,
    )
    return engineered


def save_score(name: str, fold: int, scores, context: str, directory: Path = CACHE):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.{fold}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, scores=np.asarray(scores, dtype=float), context=context)


def read_score(name: str, fold: int, expected_rows: int, context: str, directory: Path = CACHE):
    path = directory / f"{name}.{fold}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cached:
        scores = cached["scores"]
        if (
            cached["context"].item() != context
            or scores.shape != (expected_rows,)
            or not np.isfinite(scores).all()
            or not ((0 <= scores) & (scores <= 1)).all()
        ):
            raise DataValidationError(f"Invalid or stale scores in {path.name}.")
        return scores
