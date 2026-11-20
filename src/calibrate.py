"""Cross-fit sigmoid calibration and choose a policy on training scores."""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from src.artifacts import frame_csv, json_bytes, training_frame_sha256
from src.compare_models import load_selected_model
from src.cross_validate import FOLDS_PATH, N_FOLDS, RESULTS_DIR, ModelConfiguration, load_folds, training_with_folds
from src.data_contract import FEATURE_COLUMNS, ID_COLUMN, SPLIT_SEED, DataValidationError, write_new_files
from src.metrics import operating_metrics, precision_recall_target, probability_metrics
from src.policy import (
    COST_SCENARIO_PATH, ContactPolicy, CostScenario, contact_decisions, customer_costs, load_cost_scenario, select_policy,
)
from src.preprocessing import churn_probabilities
from src.train_data import load_train


CALIBRATION_RECIPE = {"method": "sigmoid", "ensemble": False, "inner_folds": N_FOLDS, "shuffle": True, "seed": SPLIT_SEED}


def build_calibrated(config: ModelConfiguration) -> CalibratedClassifierCV:
    inner = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    return CalibratedClassifierCV(estimator=config.build(), method="sigmoid", ensemble=False, cv=inner)


def cross_fit_calibration(train: pd.DataFrame, folds: pd.DataFrame, config: ModelConfiguration) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered, target, membership = training_with_folds(train, folds)
    features = ordered.loc[:, list(FEATURE_COLUMNS)]
    baseline_scores = np.empty(len(ordered), dtype=float)
    calibrated_scores = np.empty(len(ordered), dtype=float)
    metrics = []
    for fold in range(N_FOLDS):
        valid = membership == fold
        y_fit = target[~valid]
        if min(np.count_nonzero(y_fit == value) for value in (0, 1)) < N_FOLDS:
            raise DataValidationError("Each outer training fold needs at least five rows of each class for inner calibration.")
        x_fit, x_valid = features.loc[~valid], features.loc[valid]
        baseline = config.build().fit(x_fit, y_fit)
        calibrated = build_calibrated(config).fit(x_fit, y_fit)
        baseline_scores[valid] = churn_probabilities(baseline, x_valid)
        calibrated_scores[valid] = churn_probabilities(calibrated, x_valid)
        before = probability_metrics(target[valid], baseline_scores[valid])
        after = probability_metrics(target[valid], calibrated_scores[valid])
        metrics.append({
            "fold": fold, "customers": int(valid.sum()),
            "baseline_brier": before["brier"], "calibrated_brier": after["brier"],
            "brier_difference": after["brier"] - before["brier"],
            "baseline_ap": before["average_precision"], "calibrated_ap": after["average_precision"],
            "ap_difference": after["average_precision"] - before["average_precision"],
        })
    oof = pd.DataFrame({ID_COLUMN: ordered[ID_COLUMN], "fold": membership, "y_true": target, "p_baseline": baseline_scores, "p_churn": calibrated_scores})
    return oof, pd.DataFrame(metrics)


def _reliability(oof: pd.DataFrame) -> tuple[pd.DataFrame, bytes]:
    target = oof["y_true"].to_numpy()
    records = []
    fig = Figure(figsize=(8, 7), layout="constrained")
    rates, counts = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])
    rates.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", label="Perfect calibration")
    for index, (label, column, color) in enumerate((("Baseline", "p_baseline", "#977129"), ("Calibrated", "p_churn", "#126782"))):
        probabilities = oof[column].to_numpy()
        members = np.minimum((probabilities * 10).astype(int), 9)
        rows = []
        for lookup in range(10):
            include = members == lookup
            n = int(include.sum())
            rows.append({
                "score": label.lower(), "bin_start": lookup / 10, "bin_end": (lookup + 1) / 10,
                "customers": n, "mean_probability": float(probabilities[include].mean()) if n else None,
                "observed_churn": float(target[include].mean()) if n else None,
            })
        records.extend(rows)
        observed = [row for row in rows if row["customers"]]
        rates.plot([row["mean_probability"] for row in observed], [row["observed_churn"] for row in observed], "o-", color=color, label=label)
        counts.bar((np.arange(10) + 0.5) / 10 + (index - 0.5) * 0.037, [row["customers"] for row in rows], width=0.037, color=color, label=label)
    rates.set(title="Reliability on outer training validation folds", ylabel="Observed churn fraction", xlim=(0, 1), ylim=(0, 1))
    rates.legend()
    counts.set(xlabel="Predicted churn probability", ylabel="Customers per bin")
    counts.legend()
    image = BytesIO()
    fig.savefig(image, format="png", dpi=160)
    return pd.DataFrame(records), image.getvalue()


def _operating_point(oof: pd.DataFrame, scenario: CostScenario, selection: dict) -> dict:
    policy = ContactPolicy.from_dict(selection["policy"])
    decisions = contact_decisions(oof["p_churn"], policy)
    operating = operating_metrics(oof["y_true"], decisions)
    costs = customer_costs(oof["y_true"], decisions, scenario)
    return {**operating, "total_cost": float(costs.sum()), "per_customer_cost": float(costs.mean())}


def write_calibration(
    train: pd.DataFrame, folds: pd.DataFrame, config: ModelConfiguration, scenario: CostScenario, output_dir: Path,
) -> dict:
    oof, paired = cross_fit_calibration(train, folds, config)
    selected = select_policy(oof, scenario)
    operating_point = _operating_point(oof, scenario, selected)
    bins, reliability = _reliability(oof)
    brier_difference = float(paired["brier_difference"].mean())
    ap_difference = float(paired["ap_difference"].mean())
    sensitivity = []
    for retention in (0.25, 0.5, 1.0):
        for ratio in (0.05, 0.10, 0.20):
            alternate = CostScenario(scenario.lost_value * ratio, scenario.lost_value, retention)
            choice = select_policy(oof, alternate)
            op = _operating_point(oof, alternate, choice)
            sensitivity.append({
                "contact_to_loss_ratio": ratio, **alternate.as_dict(), "method": choice["method"], **choice["policy"],
                "contacts": op["contacted"], "precision": op["precision"], "recall": op["recall"],
                "total_cost": op["total_cost"], "per_customer_cost": op["per_customer_cost"],
                "mean_paired_saving": choice["mean_per_customer_saving"], "standard_error": choice["standard_error"],
                **{f"fold_{row['fold']}_saving": row["per_customer_saving"] for row in choice["paired_savings"]},
            })
    report = {
        "configuration": config.as_dict(), "train_frame_sha256": training_frame_sha256(train), "recipe": CALIBRATION_RECIPE,
        "mean_paired_metrics": {
            "brier_difference": brier_difference, "ap_difference": ap_difference,
            "brier_deteriorated": brier_difference > 0, "ap_deteriorated": ap_difference < 0,
        },
        "primary_scenario": scenario.as_dict(), "policy_selection": selected, "operating_point": operating_point,
        "exploratory_pr_target": precision_recall_target(oof["y_true"], oof["p_churn"]),
        "precision_status": "undefined (zero contacts)" if operating_point["precision"] is None else "defined",
        "interpretation": "Outer training validation scores are development estimates. The policy check reuses training information across fold blocks.",
    }
    write_new_files({
        output_dir / "oof.csv": frame_csv(oof), output_dir / "paired_metrics.csv": frame_csv(paired),
        output_dir / "reliability.csv": frame_csv(bins), output_dir / "reliability.png": reliability,
        output_dir / "sensitivity.csv": frame_csv(pd.DataFrame(sensitivity)), output_dir / "report.json": json_bytes(report),
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, default=FOLDS_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--scenario", type=Path, default=COST_SCENARIO_PATH)
    args = parser.parse_args()
    try:
        train = load_train()
        config = load_selected_model(train, args.results_dir / "selected_model.json")
        report = write_calibration(train, load_folds(train, args.folds), config, load_cost_scenario(args.scenario), args.results_dir / "calibration")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Calibration failed: {error}\n")
    mean = report["mean_paired_metrics"]
    print(f"Saved training calibration. Mean Brier change {mean['brier_difference']:+.6f}; AP change {mean['ap_difference']:+.6f}.")


if __name__ == "__main__":
    main()
