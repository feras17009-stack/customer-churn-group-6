"""Produce training-only EDA tables and a churn-rate figure."""

import argparse
from io import BytesIO
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src.data_contract import FEATURE_COLUMNS, PROJECT_ROOT, TARGET_COLUMN, write_new_files
from src.train_data import load_train


def summarize_training(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tenure = pd.to_numeric(train["tenure"], errors="raise")
    tenure_group = pd.cut(tenure, [-1, 0, 12, 24, 48, float("inf")], labels=["0", "1–12", "13–24", "25–48", "49+"])
    by_tenure = train.groupby(tenure_group.rename("tenure_months"), observed=True)[TARGET_COLUMN]
    by_contract = train.groupby("Contract", dropna=False)[TARGET_COLUMN]
    tenure_table = by_tenure.agg(customers="size", churned="sum", prevalence="mean").reset_index()
    contract_table = by_contract.agg(customers="size", churned="sum", prevalence="mean").reset_index()

    predictors = train.loc[:, list(FEATURE_COLUMNS)].replace(r"^\s*$", pd.NA, regex=True)
    missing = predictors.isna().sum().rename_axis("field").reset_index(name="missing")
    missing["customers"] = len(train)
    missing["missing_fraction"] = missing["missing"] / len(train)
    charge_columns = ["MonthlyCharges", "TotalCharges"]
    charges = predictors[charge_columns].apply(pd.to_numeric, errors="raise").describe().T
    charges.index.name = "field"
    charges = charges.rename(columns={"count": "observed_customers"}).reset_index()
    charges["observed_customers"] = charges["observed_customers"].astype(int)
    charges["customers"] = len(train)
    return {
        "prevalence": pd.DataFrame({"customers": [len(train)], "churned": [train[TARGET_COLUMN].sum()], "prevalence": [train[TARGET_COLUMN].mean()]}),
        "tenure": tenure_table,
        "contract": contract_table,
        "charges": charges,
        "missingness": missing,
    }


def write_eda(train: pd.DataFrame, output_dir: Path) -> dict[str, pd.DataFrame]:
    tables = summarize_training(train)
    fig = Figure(figsize=(10, 4), layout="constrained")
    axes = fig.subplots(1, 2)
    for axis, name, column, color in (
        (axes[0], "tenure", "tenure_months", "#126782"),
        (axes[1], "contract", "Contract", "#186A52"),
    ):
        table = tables[name]
        bars = axis.bar(table[column].astype(str), table["prevalence"], color=color)
        axis.set_title("Tenure in months" if name == "tenure" else "Contract")
        axis.set_ylabel("Observed churn rate")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_ylim(0, max(0.15, table["prevalence"].max() * 1.22))
        axis.bar_label(bars, labels=[f"n={count:,}" for count in table["customers"]], padding=4, fontsize=8)
    fig.suptitle("Customer churn in the training sample")
    chart = BytesIO()
    fig.savefig(chart, format="png", dpi=150)
    files = {output_dir / f"{name}.csv": table.to_csv(index=False, lineterminator="\n").encode("utf-8") for name, table in tables.items()}
    files[output_dir / "churn_by_tenure_and_contract.png"] = chart.getvalue()
    write_new_files(files)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "eda")
    args = parser.parse_args()
    try:
        tables = write_eda(load_train(), args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Training EDA failed: {error}\n")
    print(f"Saved {len(tables)} tables and the churn-rate figure to {args.output_dir}.")


if __name__ == "__main__":
    main()
