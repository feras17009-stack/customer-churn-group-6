"""Compare fixed model settings and apply the one-standard-error selection rule."""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src.artifacts import frame_csv, json_bytes, read_json, training_frame_sha256
from src.cross_validate import CONFIGURATIONS, FOLDS_PATH, N_FOLDS, RESULTS_DIR, ModelConfiguration, load_folds, load_oof
from src.data_contract import DataValidationError, write_new_files
from src.train_data import load_train


def summarize_comparison(diagnostics: dict) -> tuple[pd.DataFrame, dict]:
    fold_ap = {config.key: np.asarray(diagnostics[config.key]["average_precision"]) for config in CONFIGURATIONS}
    mean_ap = {key: float(values.mean()) for key, values in fold_ap.items()}
    best = max(CONFIGURATIONS, key=lambda config: mean_ap[config.key])
    standard_error = float(fold_ap[best.key].std(ddof=1) / np.sqrt(N_FOLDS))
    cutoff = mean_ap[best.key] - standard_error
    selected = next(config for config in CONFIGURATIONS if mean_ap[config.key] >= cutoff)
    rationale = "Among configurations within one standard error of the highest mean fold AP, prefer logistic regression, random forest, XGBoost, then CatBoost. Prefer unweighted within a family."
    differences = {key: (values - fold_ap[best.key]).tolist() for key, values in fold_ap.items()}
    rows = []
    for config in CONFIGURATIONS:
        key = config.key
        roc = np.asarray(diagnostics[key]["roc_auc"])
        row = {
            "configuration": key, **config.as_dict(), "mean_ap": mean_ap[key],
            "sd_ap": float(fold_ap[key].std(ddof=1)), "mean_roc_auc": float(roc.mean()),
            "sd_roc_auc": float(roc.std(ddof=1)), "eligibility_cutoff": cutoff,
            "eligible": mean_ap[key] >= cutoff, "selected": config == selected,
            "selection_rationale": rationale if config == selected else "",
            **diagnostics[key].get("threshold_diagnostics", {}),
        }
        for fold in range(N_FOLDS):
            row[f"fold_{fold}_ap"] = fold_ap[key][fold]
            row[f"fold_{fold}_ap_difference_from_best"] = differences[key][fold]
            row[f"fold_{fold}_roc_auc"] = roc[fold]
        rows.append(row)
    return pd.DataFrame(rows), {
        "configuration": selected.as_dict(), "best_configuration": best.as_dict(),
        "selected_mean_average_precision": mean_ap[selected.key], "best_mean_average_precision": mean_ap[best.key],
        "best_standard_error": standard_error, "eligibility_cutoff": cutoff,
        "paired_fold_ap_differences_from_best": differences, "rationale": rationale,
    }


def compare_models(train: pd.DataFrame, folds: pd.DataFrame, results_dir: Path = RESULTS_DIR) -> tuple[pd.DataFrame, dict]:
    diagnostics = {}
    for config in CONFIGURATIONS:
        oof = load_oof(train, folds, config, results_dir)
        ap, roc = [], []
        for _, block in oof.groupby("fold", sort=True):
            ap.append(average_precision_score(block["y_true"], block["p_churn"]))
            roc.append(roc_auc_score(block["y_true"], block["p_churn"]))
        labels, decisions = oof["y_true"], oof["p_churn"].ge(0.5)
        diagnostics[config.key] = {
            "average_precision": ap, "roc_auc": roc,
            "threshold_diagnostics": {
                "pooled_precision_at_0_5": precision_score(labels, decisions, zero_division=np.nan),
                "pooled_recall_at_0_5": recall_score(labels, decisions),
                "pooled_f1_at_0_5": f1_score(labels, decisions),
            },
        }
    table, selected = summarize_comparison(diagnostics)
    selected["train_frame_sha256"] = training_frame_sha256(train)
    fig = Figure(figsize=(9, 5), layout="constrained")
    axis = fig.subplots()
    colors = ["#126782" if is_selected else "#aab8c0" for is_selected in table["selected"]]
    axis.barh(table["configuration"].str.replace("_", " "), table["mean_ap"], xerr=table["sd_ap"], color=colors, capsize=3)
    axis.axvline(selected["eligibility_cutoff"], color="#8a4421", linestyle="--", label="Eligibility cutoff")
    axis.set_xlabel("Mean fold average precision, with sample standard deviation")
    axis.set_title("Training cross-validation at fixed settings")
    axis.invert_yaxis()
    axis.legend(loc="lower right")
    figure = BytesIO()
    fig.savefig(figure, format="png", dpi=150)
    write_new_files({
        results_dir / "model_comparison.csv": frame_csv(table),
        results_dir / "selected_model.json": json_bytes(selected),
        results_dir / "model_comparison.png": figure.getvalue(),
    })
    return table, selected


def load_selected_model(train: pd.DataFrame, path: Path) -> ModelConfiguration:
    selection = read_json(path)
    if selection.get("train_frame_sha256") != training_frame_sha256(train):
        raise DataValidationError("Selected model was recorded for a different verified training frame.")
    for config in CONFIGURATIONS:
        if selection.get("configuration") == config.as_dict():
            return config
    raise DataValidationError("Selected model configuration is outside the fixed comparison.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, default=FOLDS_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    try:
        train = load_train()
        _, selected = compare_models(train, load_folds(train, args.folds), args.results_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Model comparison failed: {error}\n")
    print(f"Selected {selected['configuration']['model_name']} ({selected['configuration']['weighting']}).")


if __name__ == "__main__":
    main()
