import csv
from dataclasses import replace
from hashlib import sha256

import pandas as pd
import pytest

from src.data_contract import DataValidationError, DatasetPin
from src.split_data import split_source
from src.train_data import load_train, write_train_manifest


def make_pin(digest, *, source_rows=60):
    return DatasetPin(digest, source_rows, 60, 15, 2)


def split_fixture(tmp_path, write_source, rows):
    source, digest = write_source(rows)
    private = tmp_path / "private"
    release = tmp_path / "release"
    pin = make_pin(digest, source_rows=len(rows))
    split_source(source, private, release, pin=pin)
    return private, release, pin


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_manifest(release, pin):
    return write_train_manifest(
        release, asset_version="synthetic-v1",
        download_url="https://example.org/synthetic-v1/train.csv", pin=pin,
    )


def test_intake_checks_digest_before_parsing(tmp_path, write_source, source_rows):
    source, digest = write_source(source_rows)
    source.write_bytes(b"unrecognized export")
    with pytest.raises(DataValidationError, match="SHA-256"):
        split_source(source, tmp_path / "private", tmp_path / "release", pin=make_pin(digest))
    assert not (tmp_path / "private").exists()
    assert not (tmp_path / "release").exists()


def test_intake_rejects_wide_export(tmp_path, write_source, source_rows):
    header = (*source_rows[0], "Churn Score")
    source, digest = write_source(source_rows, header=header)
    with pytest.raises(DataValidationError, match="header"):
        split_source(source, tmp_path / "private", tmp_path / "release", pin=make_pin(digest))


@pytest.mark.parametrize(("index", "field", "value", "error"), [
    (0, "TotalCharges", "0", "tenure-zero"),
    (2, "TotalCharges", " \t", "tenure-zero"),
    (2, "TotalCharges", "NaN", "TotalCharges"),
    (2, "TotalCharges", "bad", "TotalCharges"),
    (2, "MonthlyCharges", "inf", "MonthlyCharges"),
    (2, "customerID", " \t", "empty customerID"),
    (2, "Churn", "Unknown", "Churn"),
])
def test_intake_rejects_invalid_records(tmp_path, write_source, source_rows, index, field, value, error):
    source_rows[index][field] = value
    source, digest = write_source(source_rows)
    with pytest.raises(DataValidationError, match=error):
        split_source(source, tmp_path / "private", tmp_path / "release", pin=make_pin(digest))


def test_duplicate_rules_preserve_distinct_customers(tmp_path, write_source, source_rows):
    source_rows[8] = {**source_rows[4], "customerID": "synthetic-008"}
    source_rows.append({**source_rows[4], "customerID": " synthetic-004 "})
    private, release, _ = split_fixture(tmp_path, write_source, source_rows)
    labeled = read_csv(private / "holdout.csv") + read_csv(release / "train.csv")
    indexed = {row.pop("row_id"): row for row in labeled}
    assert len(indexed) == 60
    assert indexed["synthetic-004"] == indexed["synthetic-008"]


def test_intake_rejects_conflicting_id(tmp_path, write_source, source_rows):
    source_rows.append({**source_rows[4], "Churn": "No"})
    source, digest = write_source(source_rows)
    with pytest.raises(DataValidationError, match="conflicting"):
        split_source(source, tmp_path / "private", tmp_path / "release", pin=make_pin(digest, source_rows=61))


def test_split_is_stratified_complete_and_order_independent(tmp_path, write_source, source_rows):
    private, release, _ = split_fixture(tmp_path, write_source, source_rows)
    source, digest = write_source(reversed(source_rows), name="reordered.csv")
    split_source(source, tmp_path / "private-reordered", tmp_path / "release-reordered", pin=make_pin(digest))
    assert (release / "splits.csv").read_bytes() == (tmp_path / "release-reordered" / "splits.csv").read_bytes()
    assert (release / "train.csv").read_bytes() == (tmp_path / "release-reordered" / "train.csv").read_bytes()

    train = read_csv(release / "train.csv")
    holdout = read_csv(private / "holdout.csv")
    split_rows = read_csv(release / "splits.csv")
    train_ids = {row["row_id"] for row in train}
    holdout_ids = {row["row_id"] for row in holdout}
    assert not train_ids & holdout_ids
    assert train_ids | holdout_ids == {row["customerID"] for row in source_rows}
    assert len(train_ids) == 48 and len(holdout_ids) == 12
    assert sum(int(row["Churn"]) for row in train) == 12
    assert sum(int(row["Churn"]) for row in holdout) == 3
    assert all(tuple(row) == ("row_id", "split") for row in split_rows)
    assert not (release / "holdout.csv").exists()
    raw_charges = {row["customerID"]: row["TotalCharges"] for row in source_rows}
    assert all(row["TotalCharges"] == raw_charges[row["row_id"]] for row in train + holdout)


def test_manifest_and_loader_use_only_release_files(tmp_path, write_source, source_rows):
    private, release, pin = split_fixture(tmp_path, write_source, source_rows)
    manifest = write_manifest(release, pin)
    (private / "holdout.csv").unlink()
    frame = load_train(release, pin=pin)
    assert manifest["raw_source_sha256"] == pin.sha256
    assert manifest["train_sha256"] == sha256((release / "train.csv").read_bytes()).hexdigest()
    assert manifest["byte_count"] == (release / "train.csv").stat().st_size
    assert manifest["row_count"] == len(frame) == 48
    assert frame["Churn"].dtype == "int64"
    assert frame["row_id"].is_monotonic_increasing
    assert frame["row_id"].is_unique


def test_loader_rejects_same_size_checksum_change(tmp_path, write_source, source_rows):
    _, release, pin = split_fixture(tmp_path, write_source, source_rows)
    write_manifest(release, pin)
    path = release / "train.csv"
    path.write_bytes(path.read_bytes().replace(b"synthetic-", b"untrusted-", 1))
    with pytest.raises(DataValidationError, match="SHA-256"):
        load_train(release, pin=pin)


def test_local_release_uses_the_same_training_checks(tmp_path, write_source, source_rows):
    _, release, pin = split_fixture(tmp_path, write_source, source_rows)
    manifest = write_train_manifest(release, asset_version="local-v1", local_only=True, pin=pin)
    assert manifest["local_only"] is True
    assert manifest["download_url"] is None
    assert len(load_train(release, pin=pin)) == 48
    (release / "train.csv").write_bytes(b"changed")
    with pytest.raises(DataValidationError, match="byte count"):
        load_train(release, pin=pin)


@pytest.mark.parametrize("tamper", ["holdout", "duplicate", "unknown"])
def test_release_rejects_ids_even_with_new_digest(tmp_path, write_source, source_rows, tamper):
    private, release, pin = split_fixture(tmp_path, write_source, source_rows)
    train_path = release / "train.csv"
    train = pd.read_csv(train_path, keep_default_na=False)
    expected = "coverage"
    if tamper == "holdout":
        train.loc[0, "row_id"] = read_csv(private / "holdout.csv")[0]["row_id"]
        expected = "holdout"
    elif tamper == "duplicate":
        train.loc[0, "row_id"] = train.loc[1, "row_id"]
        expected = "duplicate"
    else:
        train.loc[0, "row_id"] = "synthetic-unknown"
    train.to_csv(train_path, index=False)
    with pytest.raises(DataValidationError, match=expected):
        write_manifest(release, pin)


def test_failed_split_retry_keeps_existing_files(tmp_path, write_source, source_rows):
    private, release, pin = split_fixture(tmp_path, write_source, source_rows)
    old_train = (release / "train.csv").read_bytes()
    with pytest.raises(FileExistsError):
        split_source(tmp_path / "synthetic.csv", tmp_path / "new-private", release, pin=pin)
    assert (release / "train.csv").read_bytes() == old_train
    assert not (tmp_path / "new-private" / "holdout.csv").exists()
    assert (private / "holdout.csv").exists()


def test_custodian_output_directories_must_be_disjoint(tmp_path, write_source, source_rows):
    source, digest = write_source(source_rows)
    release = tmp_path / "release"
    with pytest.raises(DataValidationError, match="disjoint"):
        split_source(source, release / "private", release, pin=make_pin(digest))


def test_intake_rejects_aggregate_count_drift(tmp_path, write_source, source_rows):
    source, digest = write_source(source_rows)
    pin = replace(make_pin(digest), positives=16)
    with pytest.raises(DataValidationError, match="positive label count"):
        split_source(source, tmp_path / "private", tmp_path / "release", pin=pin)
