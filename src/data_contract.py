"""Shared dataset names, source pin and CSV validation."""

import csv
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TARGET_COLUMN = "Churn"
ID_COLUMN = "row_id"
FEATURE_COLUMNS = (
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
)
RAW_COLUMNS = ("customerID", *FEATURE_COLUMNS, TARGET_COLUMN)
LABELED_COLUMNS = (ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN)
SPLIT_COLUMNS = (ID_COLUMN, "split")
HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 42


class DataValidationError(ValueError):
    """A file failed the project data contract."""


@dataclass(frozen=True)
class DatasetPin:
    sha256: str
    source_rows: int
    unique_rows: int
    positives: int
    blank_total_charges: int

    @property
    def holdout_rows(self) -> int:
        return math.ceil(self.unique_rows * HOLDOUT_FRACTION)

    @property
    def train_rows(self) -> int:
        return self.unique_rows - self.holdout_rows


IBM_PIN = DatasetPin(
    sha256="16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91",
    source_rows=7043,
    unique_rows=7043,
    positives=1869,
    blank_total_charges=11,
)


def parse_csv(payload: bytes, columns: tuple[str, ...], name: str) -> list[dict[str, str]]:
    try:
        stream = io.StringIO(payload.decode("utf-8-sig"), newline="")
        reader = csv.reader(stream, strict=True)
        if tuple(next(reader, ())) != columns:
            raise DataValidationError(f"{name} has an unexpected CSV header.")
        rows = []
        for row in reader:
            if len(row) != len(columns):
                raise DataValidationError(f"{name} has an incorrect column count near line {reader.line_num}.")
            rows.append(dict(zip(columns, row, strict=True)))
        return rows
    except (UnicodeDecodeError, csv.Error) as error:
        raise DataValidationError(f"{name} is not a valid UTF-8 CSV.") from error


def csv_bytes(columns: tuple[str, ...], rows: Iterable[Mapping[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_new_files(files: Mapping[Path, bytes]) -> None:
    """Write a new set of files and remove this run's outputs if a write fails."""
    for path in files:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}. Use a new output location.")
    created = []
    try:
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                created.append(path)
                stream.write(content)
    except OSError:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def validate_predictors(row: Mapping[str, str], location: str) -> tuple[int, bool]:
    """Check numeric fields without altering the published predictor strings."""
    try:
        tenure = int(row["tenure"])
        if tenure < 0:
            raise ValueError
    except ValueError as error:
        raise DataValidationError(f"{location} has invalid tenure.") from error
    if row["SeniorCitizen"] not in {"0", "1"}:
        raise DataValidationError(f"{location} has invalid SeniorCitizen.")

    blank_charge = not row["TotalCharges"].strip()
    if blank_charge != (tenure == 0):
        raise DataValidationError(f"{location} violates the blank TotalCharges and tenure-zero rule.")
    for column in ("MonthlyCharges", "TotalCharges"):
        if column == "TotalCharges" and blank_charge:
            continue
        try:
            if not math.isfinite(float(row[column])):
                raise ValueError
        except ValueError as error:
            raise DataValidationError(f"{location} has invalid {column}.") from error
    return tenure, blank_charge
