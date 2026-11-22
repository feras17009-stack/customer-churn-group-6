"""Load the model's input schema and validate one customer for the demo."""

import json
import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import pandas as pd

from src.data_contract import FEATURE_COLUMNS, PROJECT_ROOT, DataValidationError


SCHEMA_PATH = PROJECT_ROOT / "models" / "input_schema.json"
SERVICE_DEPENDENCIES = {
    "MultipleLines": ("PhoneService", "No phone service"),
    **dict.fromkeys(
        ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"),
        ("InternetService", "No internet service"),
    ),
}


def parse_input_schema(payload: bytes) -> dict:
    try:
        schema = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DataValidationError("Input schema is not valid JSON.") from error
    if not isinstance(schema, dict) or schema.get("schema_version") != 1:
        raise DataValidationError("Expected version 1 of the input schema.")
    fields = schema.get("fields")
    if not isinstance(fields, list) or any(not isinstance(field, dict) for field in fields):
        raise DataValidationError("Input schema must define a list of field objects.")
    if tuple(field.get("name") for field in fields) != FEATURE_COLUMNS:
        raise DataValidationError("Input schema must list the 19 predictors in their original order.")
    for field in fields:
        kind = field.get("type")
        if kind not in ("category", "integer", "number") or not isinstance(field.get("label"), str):
            raise DataValidationError(f"Invalid type or label in input schema for {field['name']}.")
        choices = field.get("choices")
        if kind == "category" and (not isinstance(choices, list) or not choices or any(type(value) is not str for value in choices)):
            raise DataValidationError(f"Input schema needs string choices for {field['name']}.")
        if field["name"] == "SeniorCitizen" and (kind != "integer" or choices != [0, 1]):
            raise DataValidationError("SeniorCitizen must have integer choices 0 and 1.")
        if kind != "category" and choices is None:
            minimum = field.get("minimum")
            if type(minimum) not in (float, int) or not math.isfinite(minimum) or minimum < 0:
                raise DataValidationError(f"Input schema needs a nonnegative numeric minimum for {field['name']}.")
            if kind == "integer" and type(minimum) is not int:
                raise DataValidationError(f"Input schema needs an integer minimum for {field['name']}.")
    return schema


def load_input_schema(path: Path = SCHEMA_PATH) -> dict:
    return parse_input_schema(Path(path).read_bytes())


def customer_choices(field: dict, values: Mapping[str, object]) -> list:
    """Return the schema choices compatible with the selected base service."""
    choices = field["choices"]
    dependency = SERVICE_DEPENDENCIES.get(field["name"])
    if dependency is None:
        return choices
    parent, unavailable = dependency
    if values[parent] == "No":
        return [unavailable]
    return [choice for choice in choices if choice != unavailable]


def make_customer_frame(schema: dict, values: Mapping[str, object]) -> pd.DataFrame:
    if set(values) != set(FEATURE_COLUMNS):
        raise DataValidationError("A customer must contain exactly the 19 input fields.")
    record = {}
    for field in schema["fields"]:
        name, kind = field["name"], field["type"]
        value = values[name]
        if field.get("nullable") and (value is None or isinstance(value, str) and not value.strip()):
            record[name] = None
            continue
        expected_type = {"category": str, "integer": Integral, "number": Real}[kind]
        if isinstance(value, bool) or not isinstance(value, expected_type):
            raise DataValidationError(f"{name} must be a valid {kind}.")
        if "choices" in field and value not in field["choices"]:
            raise DataValidationError(f"{name} must use one of the listed choices.")
        if kind != "category" and (not math.isfinite(value) or value < field.get("minimum", -math.inf)):
            raise DataValidationError(f"{name} is outside its allowed range.")
        record[name] = value
    for field in schema["fields"]:
        name = field["name"]
        if name in SERVICE_DEPENDENCIES:
            parent, _ = SERVICE_DEPENDENCIES[name]
            choices = customer_choices(field, record)
            if record[name] not in choices:
                allowed = " or ".join(choices)
                raise DataValidationError(f"{name} must be {allowed} when {parent} is {record[parent]}.")
    return pd.DataFrame([record], columns=FEATURE_COLUMNS)
