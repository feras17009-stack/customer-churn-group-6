import csv
from hashlib import sha256

import pytest


SOURCE_HEADER = (
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
)


@pytest.fixture
def source_rows():
    """Create fictional customers independently of the IBM dataset and app code."""
    rows = []
    for i in range(60):
        tenure = 0 if i < 2 else i
        monthly = 24.5 + (i % 7) * 3.25
        internet = ["DSL", "Fiber optic", "No"][i % 3]
        services = "No internet service" if internet == "No" else "No"
        values = (
            f"synthetic-{i:03}", "Female" if i % 2 else "Male", str(i % 2),
            "Yes" if i % 4 else "No", "No", str(tenure), "Yes", "No", internet,
            services, services, services, services, services, services,
            ["Month-to-month", "One year", "Two year"][i % 3], "Yes",
            "Electronic check" if i % 5 else "Mailed check", str(monthly),
            str(monthly * tenure) if tenure else "  ", "Yes" if i % 4 == 0 else "No",
        )
        rows.append(dict(zip(SOURCE_HEADER, values, strict=True)))
    return rows


@pytest.fixture
def write_source(tmp_path):
    def write(rows, *, name="synthetic.csv", header=SOURCE_HEADER):
        source = tmp_path / name
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return source, sha256(source.read_bytes()).hexdigest()

    return write


@pytest.fixture
def training_release(tmp_path, write_source, source_rows):
    from src.data_contract import DatasetPin
    from src.split_data import split_source
    from src.train_data import load_train, write_train_manifest

    source, digest = write_source(source_rows)
    pin = DatasetPin(digest, 60, 60, 15, 2)
    private, release = tmp_path / "private", tmp_path / "release"
    split_source(source, private, release, pin=pin)
    write_train_manifest(release, asset_version="synthetic-v1", download_url="https://example.org/train.csv", pin=pin)
    return load_train(release, pin=pin), release, private, pin
