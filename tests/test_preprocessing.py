import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from src.data_contract import FEATURE_COLUMNS
from src.input_schema import load_input_schema, make_customer_frame
from src.preprocessing import build_pipeline, churn_probabilities


@pytest.fixture
def examples(source_rows):
    features = pd.DataFrame(source_rows, columns=FEATURE_COLUMNS)
    return features, np.array([int(row["Churn"] == "Yes") for row in source_rows])


@pytest.mark.parametrize("model", ["logistic_regression", "catboost"])
@pytest.mark.parametrize(("tenure", "observed"), [(0, "  "), (5, "\t"), (0, "19.50"), (5, "19.50")])
def test_charge_rules_and_probability_after_reload(examples, model, tenure, observed, tmp_path):
    features, target = examples
    pipeline = build_pipeline(model).fit(features, target)
    customer = features.iloc[[0]].assign(tenure=tenure, TotalCharges=observed, Contract="New plan")
    numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    if model == "logistic_regression":
        numeric = numeric.named_steps["charges"]
    transformed = numeric.transform(customer.loc[:, ["tenure", "MonthlyCharges", "TotalCharges"]])
    expected = float(observed) if observed.strip() else (0 if tenure == 0 else features["TotalCharges"].replace("  ", np.nan).astype(float).median())
    assert transformed["TotalCharges"].iloc[0] == expected

    path = tmp_path / "fixture_model.joblib"
    joblib.dump(pipeline, path)
    probability = churn_probabilities(pipeline, customer)
    assert 0 <= probability[0] <= 1
    np.testing.assert_array_equal(churn_probabilities(joblib.load(path), customer), probability)


@pytest.mark.parametrize("model", ["logistic_regression", "random_forest", "xgboost", "catboost"])
@pytest.mark.parametrize("weighting", ["none", "balanced"])
def test_clone_classifier_contract(examples, model, weighting):
    features, target = examples
    original = build_pipeline(model, weighting=weighting)
    estimator = clone(original)
    assert is_classifier(estimator)
    fitted = estimator.fit(features, target)
    np.testing.assert_array_equal(fitted.classes_, [0, 1])
    assert not hasattr(original.named_steps["classifier"], "classes_")
    test = features.iloc[[0]].assign(Contract="Unseen contract", Partner=None, SeniorCitizen=1.0)
    assert fitted.predict_proba(test).shape == (1, 2)
    if model == "catboost":
        transformed = fitted.named_steps["preprocess"].transform(test)
        assert isinstance(transformed, pd.DataFrame)
        assert list(transformed.columns) == list(fitted.named_steps["classifier"].feature_names_in_)
        assert transformed["Partner"].iloc[0] == "__MISSING__"
        assert transformed["SeniorCitizen"].iloc[0] == "1"


def test_xgboost_balancing_is_recomputed_for_every_fit(examples, monkeypatch):
    features, target = examples
    from src.preprocessing import XGBoostClassifier

    calls = []
    real_fit = XGBoostClassifier.fit

    def record(self, X, y):
        result = real_fit(self, X, y)
        expected = np.count_nonzero(y == 0) / np.count_nonzero(y == 1)
        actual = self.estimator_.get_params()["scale_pos_weight"]
        calls.append((len(y), expected, actual))
        return result

    monkeypatch.setattr(XGBoostClassifier, "fit", record)
    estimator = build_pipeline("xgboost", weighting="balanced")
    estimator.fit(features, target)
    more_positives = target.copy()
    more_positives[[1, 2]] = 1
    estimator.fit(features, more_positives)
    assert calls[0][2] != calls[1][2]
    calibrated = CalibratedClassifierCV(
        estimator=estimator, method="sigmoid", ensemble=False,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
    ).fit(features, more_positives)
    assert len(calls) == 8
    assert all(size == 48 for size, _, _ in calls[2:7])
    assert calls[-1][0] == 60
    assert all(actual == expected for _, expected, actual in calls)
    assert calibrated.predict_proba(features.iloc[:1]).shape == (1, 2)
    plain = build_pipeline("xgboost").fit(features, target)
    assert plain.named_steps["classifier"].estimator_.get_params()["scale_pos_weight"] == 1


def test_schema_validates_app_fields_and_allows_missing_charge(examples):
    features, _ = examples
    schema = load_input_schema()
    values = features.iloc[0].to_dict()
    values.update(SeniorCitizen=0, tenure=0, MonthlyCharges=25.0, TotalCharges=None)
    customer = make_customer_frame(schema, values)
    assert tuple(customer.columns) == FEATURE_COLUMNS
    assert customer["TotalCharges"].isna().all()
    with pytest.raises(ValueError, match="Contract"):
        make_customer_frame(schema, {**values, "Contract": "Unknown"})
    with pytest.raises(ValueError, match="tenure"):
        make_customer_frame(schema, {**values, "tenure": -1})
    with pytest.raises(ValueError, match="SeniorCitizen"):
        make_customer_frame(schema, {**values, "SeniorCitizen": True})
