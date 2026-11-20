import pandas as pd

from src.data_contract import FEATURE_COLUMNS
from src.eda import summarize_training, write_eda


def test_eda_tracks_counts_and_whitespace_missingness(source_rows, tmp_path):
    training = pd.DataFrame(source_rows).assign(Churn=lambda frame: frame["Churn"].eq("Yes").astype(int))
    result = summarize_training(training)
    assert result["prevalence"].iloc[0].to_dict() == {"customers": 60, "churned": 15, "prevalence": 0.25}
    assert result["missingness"].set_index("field").loc["TotalCharges", "missing"] == 2
    assert len(result["missingness"]) == len(FEATURE_COLUMNS)
    for group in ("tenure", "contract"):
        assert result[group]["customers"].sum() == 60
        assert result[group]["churned"].sum() == 15
    write_eda(training, tmp_path)
    assert (tmp_path / "churn_by_tenure_and_contract.png").read_bytes().startswith(b"\x89PNG")
