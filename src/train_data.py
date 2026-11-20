"""Create or verify a versioned training-only release."""

import argparse
import json
import re
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from src.data_contract import (
    DATA_DIR, IBM_PIN, ID_COLUMN, LABELED_COLUMNS, SPLIT_COLUMNS, TARGET_COLUMN,
    DatasetPin, DataValidationError, parse_csv, validate_predictors, write_new_files,
)


def _validate_manifest(manifest: object, pin: DatasetPin) -> None:
    if not isinstance(manifest, dict):
        raise DataValidationError("Training manifest must contain a JSON object.")
    version = manifest.get("asset_version")
    if not isinstance(version, str) or not version.strip() or any(char.isspace() for char in version):
        raise DataValidationError("Training manifest requires a nonempty asset_version without spaces.")
    url = manifest.get("download_url")
    try:
        if not isinstance(url, str) or any(char.isspace() for char in url):
            raise ValueError
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username is not None or parsed.fragment:
            raise ValueError
        _ = parsed.port
    except ValueError as error:
        raise DataValidationError("Training manifest requires a stable HTTPS download_url without embedded credentials.") from error

    if type(manifest.get("byte_count")) is not int or manifest["byte_count"] < 1:
        raise DataValidationError("Training manifest byte_count must be a positive integer.")
    if type(manifest.get("row_count")) is not int or manifest["row_count"] != pin.train_rows:
        raise DataValidationError(f"Training manifest must declare {pin.train_rows} training rows.")
    if manifest.get("raw_source_sha256") != pin.sha256:
        raise DataValidationError("Training manifest does not reference the pinned source SHA-256.")
    digest = manifest.get("train_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise DataValidationError("Training manifest contains an invalid train SHA-256.")


def partition_ids(payload: bytes, pin: DatasetPin) -> tuple[set[str], set[str]]:
    rows = parse_csv(payload, SPLIT_COLUMNS, "Split manifest")
    partitions: dict[str, set[str]] = {"train": set(), "holdout": set()}
    seen = set()
    for index, row in enumerate(rows, start=2):
        row_id = row[ID_COLUMN]
        if not row_id or row_id != row_id.strip() or row_id in seen:
            raise DataValidationError(f"Split manifest line {index} has a duplicate or invalid row_id.")
        partition = row["split"]
        if partition not in partitions:
            raise DataValidationError(f"Split manifest line {index} has an invalid split assignment.")
        seen.add(row_id)
        partitions[partition].add(row_id)
    if len(partitions["train"]) != pin.train_rows or len(partitions["holdout"]) != pin.holdout_rows:
        raise DataValidationError("Split manifest partition sizes differ from the dataset pin.")
    return partitions["train"], partitions["holdout"]


def _validated_train_rows(payload: bytes, data_dir: Path, pin: DatasetPin) -> list[dict[str, str]]:
    train_ids, holdout_ids = partition_ids((data_dir / "splits.csv").read_bytes(), pin)
    rows = parse_csv(payload, LABELED_COLUMNS, "Training CSV")
    seen = set()
    for index, row in enumerate(rows, start=2):
        row_id = row[ID_COLUMN]
        if row_id in holdout_ids:
            raise DataValidationError(f"Training CSV line {index} contains a holdout row_id.")
        if row_id in seen:
            raise DataValidationError(f"Training CSV line {index} contains a duplicate row_id.")
        seen.add(row_id)
        validate_predictors(row, f"Training CSV line {index}")
        if row[TARGET_COLUMN] not in {"0", "1"}:
            raise DataValidationError(f"Training CSV line {index} has a Churn label outside 0 and 1.")
    if seen != train_ids:
        raise DataValidationError("Training row_id coverage differs from the split manifest.")
    return sorted(rows, key=lambda row: row[ID_COLUMN])


def write_train_manifest(
    data_dir: Path, *, asset_version: str, download_url: str, pin: DatasetPin = IBM_PIN,
) -> dict[str, str | int]:
    """Record the approved training file and its immutable download location."""
    data_dir = Path(data_dir)
    payload = (data_dir / "train.csv").read_bytes()
    rows = _validated_train_rows(payload, data_dir, pin)
    manifest = {
        "asset_version": asset_version, "download_url": download_url, "byte_count": len(payload),
        "row_count": len(rows), "raw_source_sha256": pin.sha256, "train_sha256": sha256(payload).hexdigest(),
    }
    _validate_manifest(manifest, pin)
    write_new_files({data_dir / "train_manifest.json": (json.dumps(manifest, indent=2) + "\n").encode("utf-8")})
    return manifest


def load_train(data_dir: str | Path = DATA_DIR, *, pin: DatasetPin = IBM_PIN) -> pd.DataFrame:
    """Read just the verified training release in row_id order."""
    data_dir = Path(data_dir)
    try:
        manifest = json.loads((data_dir / "train_manifest.json").read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataValidationError("Training manifest is not valid UTF-8 JSON.") from error
    _validate_manifest(manifest, pin)
    payload = (data_dir / "train.csv").read_bytes()
    if len(payload) != manifest["byte_count"]:
        raise DataValidationError("Training CSV byte count differs from the training manifest.")
    if sha256(payload).hexdigest() != manifest["train_sha256"]:
        raise DataValidationError("Training CSV SHA-256 differs from the training manifest.")
    rows = _validated_train_rows(payload, data_dir, pin)
    return pd.DataFrame(rows, columns=LABELED_COLUMNS).astype({
        "SeniorCitizen": "int64", "tenure": "int64", "MonthlyCharges": "float64", TARGET_COLUMN: "int64",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="Verify the local training release")
    verify.add_argument("--data-dir", type=Path, default=DATA_DIR)
    write = commands.add_parser("write-manifest", help="Record the published training release")
    write.add_argument("--release-dir", required=True, type=Path)
    write.add_argument("--asset-version", required=True)
    write.add_argument("--download-url", required=True)
    args = parser.parse_args()
    try:
        if args.command == "verify":
            train = load_train(args.data_dir)
            print(f"Verified {len(train)} training rows with exact training ID coverage.")
        else:
            manifest = write_train_manifest(
                args.release_dir, asset_version=args.asset_version, download_url=args.download_url,
            )
            print(f"Wrote {args.release_dir / 'train_manifest.json'} for {manifest['asset_version']}.")
    except (OSError, DataValidationError) as error:
        parser.exit(1, f"Training release check failed: {error}\n")


if __name__ == "__main__":
    main()
