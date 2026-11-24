"""Test fold-local stacking, subgroup encoding and emphasis near the recall boundary."""

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler, TargetEncoder
from threadpoolctl import threadpool_limits

from src.cross_validate import N_FOLDS
from src.data_contract import SPLIT_SEED
from src.preprocessing import CATEGORICAL_COLUMNS, churn_probabilities
from src.tabpfn_study import threshold_summary
from src.tune_classical import candidates
from src.tuning_data import alternatives, read_score, save_score, training_inputs


EXTRA = (
    ("lr_subgroup_encoding_c0.15", "Logistic regression", 0.15),
    ("lr_subgroup_encoding_c1", "Logistic regression", 1.0),
    ("xgb_frontier_weight2", "XGBoost", 2.0),
    ("xgb_frontier_weight4", "XGBoost", 4.0),
    ("stack_ridge_model_scores", "Stack", "scores"),
    ("stack_ridge_account_context", "Stack", "context"),
)
STACK_RESULTS = ("stack_ridge_model_scores", "stack_ridge_account_context")


def subgroup_features(x):
    features = x.copy()
    tenure = pd.cut(x["tenure"], [-1, 2, 6, 12, 24, 48, np.inf], labels=False).astype(str)
    contract = x["Contract"].astype(str)
    payment = x["PaymentMethod"].astype(str)
    internet = x["InternetService"].astype(str)
    features["group_contract_payment"] = contract + " | " + payment
    features["group_contract_tenure"] = contract + " | " + tenure
    features["group_internet_tenure"] = internet + " | " + tenure
    features["group_contract_internet_tenure"] = contract + " | " + internet + " | " + tenure
    features["group_billing_tenure"] = payment + " | " + x["PaperlessBilling"].astype(str) + " | " + tenure
    return features


def subgroup_model(x, strength):
    groups = [name for name in x.columns if name.startswith("group_")]
    numeric = [name for name in x.columns if name not in (*CATEGORICAL_COLUMNS, *groups)]
    transforms = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ("spline", SplineTransformer(n_knots=5, degree=2)),
        ]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_COLUMNS)),
        ("subgroups", TargetEncoder(target_type="binary", smooth=15.0, cv=4, random_state=SPLIT_SEED), groups),
    ])
    return Pipeline([("features", transforms), ("fit", LogisticRegression(C=strength, max_iter=1000))])


def model_scores(build, x, y, validation, splits):
    inner = np.empty(len(y), dtype=float)
    outer = None
    for fit, test in splits:
        model = build().fit(x.iloc[fit], y[fit])
        inner[test] = churn_probabilities(model, x.iloc[test])
    if validation is not None:
        model = build().fit(x, y)
        outer = churn_probabilities(model, validation)
    return inner, outer


def stack_features(matrix, x, context):
    logits = logit(np.clip(matrix, 1e-6, 1 - 1e-6))
    features = pd.DataFrame(logits, index=x.index, columns=[f"score_{i}" for i in range(logits.shape[1])])
    if context:
        short_tenure = np.log1p(x["tenure"])
        features["log_tenure"] = short_tenure
        features["monthly_charge"] = x["MonthlyCharges"]
        features["monthly_contract"] = x["Contract"].eq("Month-to-month").astype(int)
        features["fiber"] = x["InternetService"].eq("Fiber optic").astype(int)
        for index in range(logits.shape[1]):
            features[f"score_x_tenure_{index}"] = logits[:, index] * short_tenure
    return features


def main():
    _, _, raw, labels, membership, context = training_inputs()
    derived = alternatives(raw)
    subgroups = subgroup_features(derived)
    methods = {candidate.name: candidate for candidate in candidates(raw, derived)}
    stack_names = (
        "lr_raw_spline_c0.2", "rf_derived_d8_l12_f0.6_None",
        "xgb_derived_d2_n450_w1", "cat_raw_d4_n400_w1",
    )
    probabilities = {key: np.empty(len(labels), dtype=float) for key, _, _ in EXTRA}
    with threadpool_limits(limits=1):
        for fold in range(N_FOLDS):
            valid = membership == fold
            y = labels[~valid]
            data = {
                key: read_score(key, fold, int(valid.sum()), context)
                for key, _, _ in EXTRA
            }
            if any(data[key] is None for key in STACK_RESULTS):
                splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SPLIT_SEED + fold).split(raw.loc[~valid], y))
                inner, outer = [], []
                for name in stack_names:
                    spec = methods[name]
                    table = derived if spec.derived else raw
                    training, validation = model_scores(spec.build, table.loc[~valid], y, table.loc[valid], splits)
                    inner.append(training)
                    outer.append(validation)
                for key, mode in (("stack_ridge_model_scores", False), ("stack_ridge_account_context", True)):
                    if data[key] is None:
                        estim = Pipeline([
                            ("scale", StandardScaler()),
                            ("fit", LogisticRegression(C=0.25, max_iter=1000)),
                        ])
                        train_meta = stack_features(np.column_stack(inner), raw.loc[~valid], mode)
                        test_meta = stack_features(np.column_stack(outer), raw.loc[valid], mode)
                        data[key] = churn_probabilities(estim.fit(train_meta, y), test_meta)
                        save_score(key, fold, data[key], context)
                frontier_scores = inner[2]
            elif any(data[name] is None for name in ("xgb_frontier_weight2", "xgb_frontier_weight4")):
                splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SPLIT_SEED + fold).split(derived.loc[~valid], y))
                frontier_scores, _ = model_scores(methods[stack_names[2]].build, derived.loc[~valid], y, None, splits)
            else:
                frontier_scores = None

            for key, family, setting in EXTRA:
                if data[key] is None and family == "Logistic regression":
                    model = subgroup_model(subgroups, setting).fit(subgroups.loc[~valid], y)
                    data[key] = churn_probabilities(model, subgroups.loc[valid])
                    save_score(key, fold, data[key], context)
                elif data[key] is None and family == "XGBoost":
                    center = np.quantile(frontier_scores[y == 1], 0.2)
                    distance = (frontier_scores - center) / 0.12
                    weight = 1 + setting * np.exp(-0.5 * distance ** 2)
                    model = methods[stack_names[2]].build().fit(derived.loc[~valid], y, fit__sample_weight=weight)
                    data[key] = churn_probabilities(model, derived.loc[valid])
                    save_score(key, fold, data[key], context)
                probabilities[key][valid] = data[key]
            print(f"Completed operating-point experiments on fold {fold + 1}/{N_FOLDS}.", flush=True)
    for key, _, _ in EXTRA:
        point = threshold_summary(labels, probabilities[key])["highest_precision_with_recall_80"]
        print(f"{key}: precision {point['precision']:.2%} at recall {point['recall']:.2%}.", flush=True)


if __name__ == "__main__":
    main()
