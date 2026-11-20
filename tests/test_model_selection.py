import numpy as np
import pandas as pd
import pytest

from src.compare_models import compare_models, summarize_comparison
from src.cross_validate import CONFIGURATIONS, load_folds, make_folds, score_configuration, write_folds


@pytest.fixture
def training(source_rows):
    frame = pd.DataFrame(source_rows).rename(columns={"customerID": "row_id"})
    frame["Churn"] = frame["Churn"].eq("Yes").astype("int64")
    return frame


def test_fold_assignments_are_exact_and_order_independent(training, tmp_path):
    path = tmp_path / "folds.csv"
    folds = write_folds(training, path)
    pd.testing.assert_frame_equal(folds, make_folds(training.sample(frac=1, random_state=3)))
    assert folds["row_id"].is_unique
    assert folds["fold"].value_counts().to_dict() == {i: 12 for i in range(5)}
    pd.testing.assert_frame_equal(folds, load_folds(training, path))
    folds.loc[0, "fold"] = (folds.loc[0, "fold"] + 1) % 5
    folds.to_csv(path, index=False)
    with pytest.raises(ValueError, match="membership"):
        load_folds(training, path)


def test_all_eight_configurations_score_identical_validation_rows(training, tmp_path):
    folds = make_folds(training)
    for configuration in CONFIGURATIONS:
        predictions = score_configuration(training, folds, configuration, tmp_path)
        assert predictions["row_id"].is_unique
        np.testing.assert_array_equal(predictions["fold"], folds["fold"])
        np.testing.assert_array_equal(predictions["y_true"], training["Churn"])
        assert predictions["p_churn"].between(0, 1).all()
    table, selection = compare_models(training, folds, tmp_path)
    assert len(table) == 8
    assert table["selected"].sum() == 1
    assert selection["configuration"] in [config.as_dict() for config in CONFIGURATIONS]
    assert (tmp_path / "selected_model.json").is_file()
    assert (tmp_path / "model_comparison.csv").is_file()
    assert (tmp_path / "model_comparison.png").read_bytes().startswith(b"\x89PNG")


def test_one_standard_error_prefers_simpler_unweighted_model():
    diagnostics = {}
    for config in CONFIGURATIONS:
        ap = [0.60] * 5
        if config.model_name == "logistic_regression":
            ap = ([0.744] if config.weighting == "none" else [0.748]) * 5
        if config.model_name == "catboost" and config.weighting == "balanced":
            ap = [0.74, 0.76, 0.75, 0.77, 0.73]
        diagnostics[config.key] = {"average_precision": ap, "roc_auc": [0.8] * 5}
    table, selection = summarize_comparison(diagnostics)
    assert selection["best_configuration"]["model_name"] == "catboost"
    assert selection["configuration"] == CONFIGURATIONS[0].as_dict()
    best = np.array([0.74, 0.76, 0.75, 0.77, 0.73])
    assert selection["eligibility_cutoff"] == pytest.approx(best.mean() - best.std(ddof=1) / np.sqrt(5))
    selected = table.loc[table["selected"]].iloc[0]
    assert selected["fold_0_ap_difference_from_best"] == pytest.approx(0.004)
