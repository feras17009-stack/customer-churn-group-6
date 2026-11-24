"""Screen fold-local feature and parameter changes for the four classical families."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from catboost import CatBoostClassifier
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from src.cross_validate import N_FOLDS
from src.data_contract import SPLIT_SEED
from src.preprocessing import CATEGORICAL_COLUMNS, churn_probabilities
from src.tabpfn_study import threshold_summary
from src.tuning_data import alternatives, read_score, save_score, training_inputs


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    derived: bool
    build: Callable[[], BaseEstimator]


def preprocessor(x, numeric_steps=(), sparse=0.3):
    numeric = [name for name in x.columns if name not in CATEGORICAL_COLUMNS]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), *numeric_steps]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_COLUMNS)),
    ], sparse_threshold=sparse)


def candidates(raw, derived):
    items = []
    for use_derived, x in ((False, raw), (True, derived)):
        table = "derived" if use_derived else "raw"
        for approach in ("linear", "spline", "interactions"):
            for strength in (0.03, 0.2, 1.0):
                def logistic(approach=approach, strength=strength, x=x):
                    numeric = [("scale", StandardScaler())]
                    if approach != "linear":
                        numeric.append(("splines", SplineTransformer(n_knots=5, degree=2)))
                    steps = [("features", preprocessor(x, numeric, sparse=1.0))]
                    if approach == "interactions":
                        steps.append(("interactions", PolynomialFeatures(2, interaction_only=True, include_bias=False)))
                    return Pipeline([*steps, ("fit", LogisticRegression(C=strength, solver="liblinear", max_iter=1000))])

                items.append(Candidate(f"lr_{table}_{approach}_c{strength:g}", "Logistic regression", use_derived, logistic))

        for depth, leaf, features, weight in (
            (None, 4, 0.55, None), (None, 10, 0.8, None), (None, 20, 0.85, None),
            (8, 12, 0.6, None), (12, 16, 0.9, "balanced"), (9, 25, 0.8, "balanced"),
        ):
            def forest(depth=depth, leaf=leaf, features=features, weight=weight, x=x):
                return Pipeline([
                    ("features", preprocessor(x)),
                    ("fit", RandomForestClassifier(
                        n_estimators=300, max_depth=depth, min_samples_leaf=leaf, max_features=features,
                        class_weight=weight, n_jobs=1, random_state=SPLIT_SEED,
                    )),
                ])
            items.append(Candidate(f"rf_{table}_d{depth}_l{leaf}_f{features:g}_{weight}", "Random forest", use_derived, forest))

        for depth, rounds, rate, penalty, min_child, weight in (
            (1, 800, 0.055, 5, 5, 1.0), (2, 450, 0.04, 4, 8, 1.0), (2, 600, 0.035, 8, 14, 2.0),
            (3, 350, 0.035, 10, 10, 1.0), (3, 500, 0.04, 14, 16, 2.6),
            (4, 250, 0.04, 6, 8, 1.0), (5, 260, 0.035, 10, 16, 1.0),
        ):
            def booster(depth=depth, rounds=rounds, rate=rate, penalty=penalty, min_child=min_child, weight=weight, x=x):
                return Pipeline([
                    ("features", preprocessor(x)),
                    ("fit", XGBClassifier(
                        n_estimators=rounds, max_depth=depth, learning_rate=rate, min_child_weight=min_child,
                        max_bin=128, reg_lambda=penalty, reg_alpha=0.2, subsample=0.85, colsample_bytree=0.85,
                        scale_pos_weight=weight, n_jobs=1, random_state=SPLIT_SEED,
                    )),
                ])
            items.append(Candidate(f"xgb_{table}_d{depth}_n{rounds}_w{weight:g}", "XGBoost", use_derived, booster))

        for depth, rounds, rate, penalty, weight in (
            (3, 450, 0.05, 6, 1.0), (4, 400, 0.04, 8, 1.0), (4, 450, 0.045, 12, 2.0),
            (5, 360, 0.04, 8, 1.0), (6, 300, 0.035, 12, 1.0),
        ):
            def cat(depth=depth, rounds=rounds, rate=rate, penalty=penalty, weight=weight):
                return CatBoostClassifier(
                    depth=depth, iterations=rounds, learning_rate=rate, l2_leaf_reg=penalty,
                    scale_pos_weight=weight, loss_function="Logloss", cat_features=list(CATEGORICAL_COLUMNS),
                    random_seed=SPLIT_SEED, verbose=False, allow_writing_files=False, thread_count=1,
                )
            items.append(Candidate(f"cat_{table}_d{depth}_n{rounds}_w{weight:g}", "CatBoost", use_derived, cat))
    return items


def screen(candidate, x, labels, membership, context):
    probabilities = np.empty(len(labels), dtype=float)
    for fold in range(N_FOLDS):
        valid = membership == fold
        scores = read_score(candidate.name, fold, int(valid.sum()), context)
        if scores is None:
            model = candidate.build().fit(x.loc[~valid], labels[~valid])
            scores = churn_probabilities(model, x.loc[valid])
            save_score(candidate.name, fold, scores, context)
        probabilities[valid] = scores
    point = threshold_summary(labels, probabilities)["highest_precision_with_recall_80"]
    print(f"{candidate.name}: precision {point['precision']:.2%} at recall {point['recall']:.2%}", flush=True)


def main():
    _, _, raw, labels, membership, context = training_inputs()
    derived = alternatives(raw)
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=4) as workers:
        queue = [
            workers.submit(screen, candidate, derived if candidate.derived else raw, labels, membership, context)
            for candidate in candidates(raw, derived)
        ]
        for future in queue:
            future.result()
    print(f"Completed {len(queue)} candidate configurations.", flush=True)


if __name__ == "__main__":
    main()
