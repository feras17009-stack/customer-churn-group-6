"""Fold-local preprocessing and sklearn-compatible churn classifiers."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.data_contract import FEATURE_COLUMNS


MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost", "catboost")
WEIGHTINGS = ("none", "balanced")
NUMERIC_COLUMNS = ("tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL_COLUMNS = tuple(column for column in FEATURE_COLUMNS if column not in NUMERIC_COLUMNS)
MISSING_CATEGORY = "__MISSING__"


class ChargeFeatures(TransformerMixin, BaseEstimator):
    """Learn the observed charge median from only the rows passed to fit."""

    @staticmethod
    def _numeric(X: pd.DataFrame) -> pd.DataFrame:
        data = X.loc[:, list(NUMERIC_COLUMNS)].copy()
        charges = data["TotalCharges"].astype("string").str.strip()
        data["TotalCharges"] = charges.mask(charges.eq(""))
        data = data.apply(pd.to_numeric, errors="raise").astype("float64")
        tenure = data["tenure"].to_numpy()
        if not np.isfinite(tenure).all() or (tenure < 0).any() or (tenure != np.floor(tenure)).any():
            raise ValueError("tenure must contain finite nonnegative integers.")
        if not np.isfinite(data["MonthlyCharges"]).all() or np.isinf(data["TotalCharges"]).any():
            raise ValueError("MonthlyCharges and observed TotalCharges must be finite.")
        return data

    def fit(self, X, y=None):
        observed = self._numeric(X)["TotalCharges"].dropna()
        self.total_charge_median_ = float(observed.median()) if not observed.empty else None
        self.feature_names_in_ = np.asarray(NUMERIC_COLUMNS, dtype=object)
        self.n_features_in_ = len(NUMERIC_COLUMNS)
        return self

    def transform(self, X):
        check_is_fitted(self)
        data = self._numeric(X)
        missing = data["TotalCharges"].isna()
        data.loc[missing & data["tenure"].eq(0), "TotalCharges"] = 0.0
        remaining = data["TotalCharges"].isna()
        if remaining.any():
            if self.total_charge_median_ is None:
                raise ValueError("Positive-tenure missing TotalCharges requires an observed training charge.")
            data.loc[remaining, "TotalCharges"] = self.total_charge_median_
        return data

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.feature_names_in_.copy()


def _categorical_frame(X: pd.DataFrame) -> pd.DataFrame:
    values = X.astype("string")
    senior = values["SeniorCitizen"].str.strip()
    values["SeniorCitizen"] = pd.to_numeric(senior.mask(senior.eq(""))).astype("Int64").astype("string")
    return values.fillna(MISSING_CATEGORY).astype(object)


class XGBoostClassifier(ClassifierMixin, BaseEstimator):
    """Set the weight from the labels received by each individual fit."""

    def __init__(self, *, weighting="none", seed=42):
        self.weighting = weighting
        self.seed = seed

    def fit(self, X, y):
        from xgboost import XGBClassifier

        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"Unsupported weighting: {self.weighting}")
        labels = np.asarray(y)
        if labels.ndim != 1 or not np.array_equal(np.unique(labels), [0, 1]):
            raise ValueError("XGBoost requires both binary churn labels 0 and 1.")
        ratio = np.count_nonzero(labels == 0) / np.count_nonzero(labels == 1)
        estimator = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, device="cpu", n_jobs=1,
            random_state=self.seed, early_stopping_rounds=None,
            scale_pos_weight=ratio if self.weighting == "balanced" else 1,
        ).fit(X, labels)
        self.estimator_ = estimator
        self.classes_ = estimator.classes_
        self.n_features_in_ = estimator.n_features_in_
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.estimator_.predict_proba(X)


class NamedCatBoostClassifier(ClassifierMixin, BaseEstimator):
    """Check that preprocessing delivers the expected named CatBoost columns."""

    def __init__(self, *, weighting="none", seed=42):
        self.weighting = weighting
        self.seed = seed

    @staticmethod
    def _validate_frame(X):
        if not isinstance(X, pd.DataFrame) or tuple(X.columns) != (*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS):
            raise ValueError("CatBoost requires the named numeric and categorical preprocessing columns.")

    def fit(self, X, y):
        from catboost import CatBoostClassifier

        self._validate_frame(X)
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"Unsupported weighting: {self.weighting}")
        estimator = CatBoostClassifier(
            iterations=200, task_type="CPU", thread_count=1, random_seed=self.seed,
            allow_writing_files=False, use_best_model=False, verbose=False,
            auto_class_weights="Balanced" if self.weighting == "balanced" else None,
            cat_features=list(CATEGORICAL_COLUMNS),
        ).fit(X, y)
        self.estimator_ = estimator
        self.classes_ = estimator.classes_
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy(copy=True)
        return self

    def predict(self, X):
        check_is_fitted(self)
        self._validate_frame(X)
        return self.estimator_.predict(X).ravel()

    def predict_proba(self, X):
        check_is_fitted(self)
        self._validate_frame(X)
        return self.estimator_.predict_proba(X)


def build_pipeline(model_name: str, *, weighting: str = "none", seed: int = 42) -> Pipeline:
    if model_name not in MODEL_NAMES or weighting not in WEIGHTINGS:
        raise ValueError(f"Unsupported model configuration: {model_name} / {weighting}")
    class_weight = "balanced" if weighting == "balanced" else None
    numeric = ChargeFeatures()
    categorical = FunctionTransformer(_categorical_frame, feature_names_out="one-to-one")
    if model_name != "catboost":
        categorical = Pipeline([("strings", categorical), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    if model_name == "logistic_regression":
        numeric = Pipeline([("charges", numeric), ("scale", StandardScaler())])
        classifier = LogisticRegression(max_iter=1000, random_state=seed, class_weight=class_weight)
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(random_state=seed, class_weight=class_weight, n_jobs=1)
    elif model_name == "xgboost":
        classifier = XGBoostClassifier(weighting=weighting, seed=seed)
    else:
        classifier = NamedCatBoostClassifier(weighting=weighting, seed=seed)
    preprocessing = ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_COLUMNS)), ("categorical", categorical, list(CATEGORICAL_COLUMNS)),
    ], verbose_feature_names_out=False)
    if model_name == "catboost":
        preprocessing.set_output(transform="pandas")
    return Pipeline([("preprocess", preprocessing), ("classifier", classifier)])


def churn_probabilities(estimator, features: pd.DataFrame) -> np.ndarray:
    positive = np.flatnonzero(np.asarray(estimator.classes_) == 1)
    if len(positive) != 1:
        raise ValueError("The fitted classifier must have one class labeled 1.")
    probabilities = np.asarray(estimator.predict_proba(features))[:, positive[0]]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Classifier returned an invalid churn probability.")
    return probabilities


def fitted_settings(pipeline: Pipeline) -> dict:
    """Record the effective settings reported by the fitted backend."""
    classifier = pipeline.named_steps["classifier"]
    if isinstance(classifier, NamedCatBoostClassifier):
        return classifier.estimator_.get_all_params()
    if isinstance(classifier, XGBoostClassifier):
        return {
            key: "NaN" if isinstance(value, float) and np.isnan(value) else value
            for key, value in classifier.estimator_.get_params(deep=False).items()
        }
    return classifier.get_params(deep=False)
