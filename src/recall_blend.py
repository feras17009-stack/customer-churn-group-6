"""The CatBoost and Random Forest blend selected under the strict recall criterion."""

import numpy as np
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

from src.data_contract import DataValidationError
from src.preprocessing import churn_probabilities
from src.tuning_data import prepare_features


BLEND_NAME = "blend_log_odds_124"
BLEND_MEMBERS = ("cat_raw_d3_n450_w1", "rf_raw_dNone_l10_f0.8_None")
BLEND_WEIGHTS = (0.25, 0.75)
BLEND_CONFIGURATION = {
    "model_name": "catboost_random_forest_blend",
    "method": "log_odds",
    "members": dict(zip(BLEND_MEMBERS, BLEND_WEIGHTS, strict=True)),
}


class RecallBlend(ClassifierMixin, BaseEstimator):
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        classes = np.unique(y)
        if not np.array_equal(classes, [0, 1]) or len(self.estimators) != len(BLEND_WEIGHTS):
            raise DataValidationError("The recall blend requires two estimators and both binary labels.")
        features = prepare_features(X)
        self.estimators_ = tuple(clone(estimator).fit(features, y) for estimator in self.estimators)
        self.classes_ = classes
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ("estimators_", "classes_"))
        features = prepare_features(X)
        all_scores = np.column_stack([churn_probabilities(estimator, features) for estimator in self.estimators_])
        positive = expit(logit(np.clip(all_scores, 1e-7, 1 - 1e-7)) @ np.asarray(BLEND_WEIGHTS))
        return np.column_stack((1 - positive, positive))

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
