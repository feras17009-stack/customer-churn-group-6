"""Consistent serialization and training provenance for run artifacts."""

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd

from src.data_contract import ID_COLUMN, LABELED_COLUMNS, DataValidationError


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def parse_json_object(payload: bytes, name: str) -> dict:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DataValidationError(f"{name} is not valid JSON.") from error
    if not isinstance(value, dict):
        raise DataValidationError(f"{name} must contain a JSON object.")
    return value


def read_json(path: Path) -> dict:
    return parse_json_object(path.read_bytes(), path.name)


def frame_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")


def training_frame_sha256(train: pd.DataFrame) -> str:
    canonical = train.loc[:, list(LABELED_COLUMNS)].sort_values(ID_COLUMN)
    return sha256(frame_csv(canonical)).hexdigest()
