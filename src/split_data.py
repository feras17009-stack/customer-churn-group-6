"""Isolated dataset intake for the data custodian."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data_contract import (
    HOLDOUT_FRACTION, IBM_PIN, ID_COLUMN, LABELED_COLUMNS, PROJECT_ROOT, RAW_COLUMNS,
    SPLIT_COLUMNS, SPLIT_SEED, TARGET_COLUMN, DatasetPin, DataValidationError,
    csv_bytes, parse_csv, validate_predictors, write_new_files,
)


def _deduplicate(raw_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    customers = {}
    for position, row in enumerate(raw_rows, start=2):
        customer_id = row.pop("customerID").strip()
        if not customer_id:
            raise DataValidationError(f"Source row {position} has an empty customerID.")
        record = {ID_COLUMN: customer_id, **row}
        if customers.setdefault(customer_id, record) != record:
            raise DataValidationError(f"Source row {position} has conflicting records for one customerID.")
    return [customers[customer_id] for customer_id in sorted(customers)]


def split_source(
    source: Path, private_dir: Path, release_dir: Path, *, pin: DatasetPin = IBM_PIN,
) -> dict[str, int | str]:
    """Validate the pinned source and write deterministic, isolated partitions."""
    source, private_dir, release_dir = (path.resolve() for path in (source, private_dir, release_dir))
    if private_dir.is_relative_to(release_dir) or release_dir.is_relative_to(private_dir):
        raise DataValidationError("The private and release directories must be disjoint.")
    if private_dir.is_relative_to(PROJECT_ROOT) or source.is_relative_to(PROJECT_ROOT):
        raise DataValidationError("Keep the private output and source CSV outside the project directory.")
    if source.is_relative_to(release_dir):
        raise DataValidationError("Keep the source CSV outside the release directory.")

    payload = source.read_bytes()
    digest = sha256(payload).hexdigest()
    if digest != pin.sha256:
        raise DataValidationError("Source SHA-256 differs from the dataset pin.")
    raw_rows = parse_csv(payload, RAW_COLUMNS, "Source")
    if len(raw_rows) != pin.source_rows:
        raise DataValidationError(f"Expected {pin.source_rows} source rows, read {len(raw_rows)}.")
    rows = _deduplicate(raw_rows)
    if len(rows) != pin.unique_rows:
        raise DataValidationError(f"Expected {pin.unique_rows} unique customer IDs, found {len(rows)}.")

    labels = []
    blank_charges = 0
    tenure_zero_rows = 0
    for position, row in enumerate(rows, start=1):
        tenure, blank_charge = validate_predictors(row, f"Unique source record {position}")
        blank_charges += blank_charge
        tenure_zero_rows += tenure == 0
        label = row[TARGET_COLUMN]
        if label not in {"Yes", "No"}:
            raise DataValidationError(f"Unique source record {position} has invalid Churn.")
        labels.append(int(label == "Yes"))
        row[TARGET_COLUMN] = str(labels[-1])
    if sum(labels) != pin.positives:
        raise DataValidationError(f"Expected a positive label count of {pin.positives}, found {sum(labels)}.")
    if blank_charges != pin.blank_total_charges:
        raise DataValidationError(f"Expected {pin.blank_total_charges} blank TotalCharges, found {blank_charges}.")

    train_indices, holdout_indices = train_test_split(
        range(len(rows)), test_size=HOLDOUT_FRACTION, stratify=labels, random_state=SPLIT_SEED,
    )
    train_set = set(train_indices)
    split_rows = (
        {ID_COLUMN: row[ID_COLUMN], "split": "train" if index in train_set else "holdout"}
        for index, row in enumerate(rows)
    )
    write_new_files({
        private_dir / "holdout.csv": csv_bytes(LABELED_COLUMNS, (rows[index] for index in sorted(holdout_indices))),
        release_dir / "train.csv": csv_bytes(LABELED_COLUMNS, (rows[index] for index in sorted(train_indices))),
        release_dir / "splits.csv": csv_bytes(SPLIT_COLUMNS, split_rows),
    })
    return {
        "raw_source_sha256": digest, "source_rows": len(raw_rows), "unique_ids": len(rows),
        "positive_labels": sum(labels), "blank_total_charges": blank_charges,
        "tenure_zero_rows": tenure_zero_rows, "train_rows": len(train_indices),
        "holdout_rows": len(holdout_indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Path to the pinned IBM CSV")
    parser.add_argument("--private-dir", required=True, type=Path, help="Isolated holdout output directory")
    parser.add_argument("--release-dir", required=True, type=Path, help="Training release output directory")
    args = parser.parse_args()
    try:
        report = split_source(args.source, args.private_dir, args.release_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Intake failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
