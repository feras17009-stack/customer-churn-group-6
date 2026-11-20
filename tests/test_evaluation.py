import json
import sys

import pytest

from src import evaluate
from src.cross_validate import CONFIGURATIONS
from src.freeze_model import write_model_bundle
from src.policy import ContactPolicy, CostScenario


def test_metrics_costs_and_paired_uncertainty():
    result = evaluate.evaluate_scores([0, 0, 1, 1], [0.1, 0.2, 0.19, 0.7], ContactPolicy("threshold", 0.2), CostScenario(1, 10, 0.5))
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["scenario_costs"]["frozen_policy"] == {"total": 17.0, "per_customer": 4.25}
    assert result["scenario_costs"]["contact_all"]["total"] == 14.0
    assert result["scenario_costs"]["contact_none"]["total"] == 20.0
    assert result["paired_savings"]["vs_contact_none"]["total"] == 3.0
    assert result["bootstrap"]["resamples"] == 2000
    intervals = result["bootstrap"]["intervals"]
    assert intervals["brier"]["valid_replicates"] == 2000
    assert 0 < intervals["roc_auc"]["valid_replicates"] < 2000
    assert result["exploratory_targets"]["reaches_both"] is False


def test_undefined_precision_and_its_interval_are_null():
    result = evaluate.evaluate_scores([0, 1], [0.05, 0.17], ContactPolicy("contact_none"), CostScenario(1, 10, 0.5))
    assert result["precision"] is None
    assert result["precision_status"] == "undefined (zero contacts)"
    assert result["bootstrap"]["intervals"]["precision"] == {"lower": None, "upper": None, "valid_replicates": 0}
    assert result["paired_savings"]["vs_contact_none"]["per_customer"] == 0


def test_custodian_cli_publishes_only_aggregate_and_refuses_repeat(training_release, tmp_path, monkeypatch):
    train, release, private, pin = training_release
    splits = release / "splits.csv"
    manifest = write_model_bundle(train, CONFIGURATIONS[0], CostScenario(1, 10, 0.5), ContactPolicy("threshold", 0.2), tmp_path / "models", splits)
    output = tmp_path / "results" / "test_metrics.json"
    args = ["evaluate", "--holdout", str(private / "holdout.csv"), "--manifest", str(manifest), "--splits", str(splits), "--output", str(output)]
    monkeypatch.setattr(evaluate, "IBM_PIN", pin)
    monkeypatch.setattr(sys, "argv", args)
    evaluate.main()
    evidence = output.read_text(encoding="utf-8")
    result = json.loads(evidence)
    assert result["freeze_id"] == "freeze-v1"
    assert result["customers"] == sum(result["confusion_matrix"].values()) == 12
    assert result["positives"] == 3
    assert "synthetic-" not in evidence and "y_true" not in evidence and "p_churn" not in evidence
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 1
    second = output.with_name("alternate.json")
    monkeypatch.setattr(sys, "argv", [*args[:-1], str(second)])
    with pytest.raises(SystemExit):
        evaluate.main()
    assert not second.exists() and output.read_text(encoding="utf-8") == evidence
