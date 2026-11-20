import numpy as np
import pandas as pd
import pytest

from src.calibrate import write_calibration
from src.cross_validate import CONFIGURATIONS, make_folds
from src.metrics import operating_metrics, precision_recall_target
from src.policy import ContactPolicy, CostScenario, contact_decisions, minimum_cost_policy, select_policy


def test_contact_boundary_constant_policies_and_zero_retention():
    scenario = CostScenario(1, 10, 0.5)
    assert scenario.break_even_policy() == ContactPolicy("threshold", 0.2)
    above = np.nextafter(0.2, 1.0)
    below = np.nextafter(0.2, 0.0)
    assert contact_decisions([below, 0.2, above], scenario.break_even_policy()).tolist() == [False, True, True]
    assert not contact_decisions([0, 1], CostScenario(1, 10, 0).break_even_policy()).any()
    assert contact_decisions([0, 1], ContactPolicy("contact_all")).all()
    assert CostScenario(1, 10, 1).break_even_policy().threshold == 0.1
    assert operating_metrics([1, 0], [False, False])["precision"] is None


def test_empirical_search_handles_tied_probabilities_and_fewer_contacts():
    scenario = CostScenario(1, 10, 0.5)
    assert minimum_cost_policy([0, 0, 0], [0.2, 0.8, 1.0], scenario) == ContactPolicy("contact_none")
    assert minimum_cost_policy([1, 1], [0.4, 0.4], scenario) == ContactPolicy("contact_all")
    assert minimum_cost_policy([1, 0, 0, 0, 0, 0], [0.5] * 5 + [0.1], scenario) == ContactPolicy("contact_none")


def test_policy_stability_checks_five_held_out_blocks():
    oof = pd.DataFrame({"fold": np.repeat(range(5), 4), "y_true": [1, 0, 0, 0] * 5, "p_churn": [0.95, 0.35, 0.07, 0.02] * 5})
    scenario = CostScenario(1, 10, 0.5)
    picked = select_policy(oof, scenario)
    assert picked["method"] == "empirical"
    assert picked["policy"] == {"policy_type": "threshold", "threshold": 0.95}
    assert [row["per_customer_saving"] for row in picked["paired_savings"]] == [0.25] * 5
    oof["p_churn"] = [0.95, 0.15, 0.07, 0.02] * 5
    assert select_policy(oof, scenario)["method"] == "break_even"


def test_recall_target_reports_attainable_precision_and_actual_recall():
    target = precision_recall_target([1, 0, 1, 0, 1], [0.9, 0.8, 0.7, 0.6, 0.5])
    assert target == {"max_precision": 0.6, "recall_at_max_precision": 1.0, "reaches_both_targets": True}


def test_calibration_writes_paired_results_and_sensitivity(source_rows, tmp_path):
    train = pd.DataFrame(source_rows).rename(columns={"customerID": "row_id"})
    train["Churn"] = train["Churn"].eq("Yes").astype("int64")
    report = write_calibration(train, make_folds(train), CONFIGURATIONS[0], CostScenario(1, 10, 0.5), tmp_path)
    paired = pd.read_csv(tmp_path / "paired_metrics.csv")
    oof = pd.read_csv(tmp_path / "oof.csv")
    bins = pd.read_csv(tmp_path / "reliability.csv")
    assert len(paired) == 5
    assert oof["row_id"].is_unique and len(oof) == len(train)
    assert set(bins.groupby("score")["customers"].sum()) == {len(train)}
    assert len(pd.read_csv(tmp_path / "sensitivity.csv")) == 9
    assert report["mean_paired_metrics"]["brier_difference"] == pytest.approx(paired["brier_difference"].mean())
    assert report["operating_point"]["customers"] == 60
    assert (tmp_path / "reliability.png").read_bytes().startswith(b"\x89PNG")
