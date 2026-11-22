import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import src.freeze_model as freeze_model
from src.app import format_decision, score_customer
from src.cross_validate import CONFIGURATIONS
from src.data_contract import PROJECT_ROOT
from src.freeze_model import load_frozen_model, verify_freeze, write_model_bundle
from src.input_schema import customer_choices
from src.policy import ContactPolicy, CostScenario, contact_decisions


@pytest.fixture
def fixture_manifest(training_release, tmp_path):
    training, release, _, _ = training_release
    return write_model_bundle(
        training, CONFIGURATIONS[0], CostScenario(1, 10, 0.5), ContactPolicy("threshold", 0.2),
        tmp_path / "models", release / "splits.csv",
    )


def test_freeze_verifies_and_reloads_one_customer(fixture_manifest):
    manifest, payloads = verify_freeze(fixture_manifest)
    assert manifest["freeze_id"] == "freeze-v1"
    assert set(payloads) == {"decision.json", "churn_model.joblib", "input_schema.json"}
    frozen = load_frozen_model(fixture_manifest)
    customer = {}
    for field in frozen.schema["fields"]:
        customer[field["name"]] = (
            customer_choices(field, customer)[0] if "choices" in field
            else None if field.get("nullable") else field["minimum"]
        )
    probability, decision = score_customer(frozen, customer)
    assert 0 <= probability <= 1
    assert decision == contact_decisions([probability], frozen.policy)[0]
    assert score_customer(load_frozen_model(fixture_manifest), customer) == (probability, decision)


def test_manifest_detects_tampering_before_loading(fixture_manifest):
    path = fixture_manifest.parent / "churn_model.joblib"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        load_frozen_model(fixture_manifest)


def test_serving_allows_only_a_python_patch_difference(fixture_manifest, monkeypatch):
    versions = freeze_model.runtime_versions()
    major, minor, patch = versions["python"].split(".")
    versions["python"] = f"{major}.{minor}.{int(patch) + 1}"
    monkeypatch.setattr(freeze_model, "runtime_versions", lambda: versions.copy())

    with pytest.raises(ValueError, match="requires the Python and library versions"):
        load_frozen_model(fixture_manifest)
    assert load_frozen_model(fixture_manifest, allow_python_patch_difference=True).decision["freeze_id"] == "freeze-v1"

    versions["python"] = f"{major}.{int(minor) + 1}.0"
    with pytest.raises(ValueError, match="requires the Python and library versions"):
        load_frozen_model(fixture_manifest, allow_python_patch_difference=True)

    versions["python"] = f"{major}.{minor}.{patch}"
    versions["scikit-learn"] = "0.0.0"
    with pytest.raises(ValueError, match="requires the Python and library versions"):
        load_frozen_model(fixture_manifest, allow_python_patch_difference=True)


@pytest.mark.parametrize(("score", "relation"), [(np.nextafter(0.2, 0), "below"), (0.2, "equal to"), (np.nextafter(0.2, 1), "above")])
def test_rounding_explanation_preserves_the_boundary(score, relation):
    rule, detail = format_decision(score, ContactPolicy("threshold", 0.2))
    assert "20.0%" in rule
    assert relation in detail


def test_streamlit_submission_scores_schema_defaults(fixture_manifest, monkeypatch):
    monkeypatch.setenv("CHURN_FREEZE_MANIFEST", str(fixture_manifest))
    app = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py"), default_timeout=20).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].label == "Estimated churn probability"
    assert app.metric[0].value.endswith("%")
    assert len(app.success) + len(app.info) == 1


def test_streamlit_service_choices_follow_base_services(fixture_manifest, monkeypatch):
    monkeypatch.setenv("CHURN_FREEZE_MANIFEST", str(fixture_manifest))
    app = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py"), default_timeout=20).run()
    assert app.selectbox(key="MultipleLines").value == "No phone service"
    app.selectbox(key="PhoneService").set_value("Yes").run()
    assert app.selectbox(key="MultipleLines").options == ["No", "Yes"]
    app.selectbox(key="MultipleLines").set_value("Yes").run()
    app.selectbox(key="OnlineSecurity").set_value("Yes").run()
    app.button[0].click().run()
    assert app.metric

    app.selectbox(key="PhoneService").set_value("No")
    app.selectbox(key="InternetService").set_value("No").run()
    assert not app.metric
    assert app.selectbox(key="MultipleLines").options == ["No phone service"]
    internet_addons = ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")
    for key in internet_addons:
        assert app.selectbox(key=key).options == ["No internet service"]
        assert app.selectbox(key=key).value == "No internet service"
    app.button[0].click().run()
    assert app.metric and not app.error and not app.exception

    app.selectbox(key="InternetService").set_value("Fiber optic").run()
    for key in internet_addons:
        assert app.selectbox(key=key).options == ["No", "Yes"]
        assert app.selectbox(key=key).value == "No"
