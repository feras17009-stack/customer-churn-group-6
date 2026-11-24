import pytest
from streamlit.testing.v1 import AppTest

from src.data_contract import PROJECT_ROOT, DataValidationError
from src.evaluate import evaluate_holdout
from src.freeze_model import APP_MANIFEST_PATH, load_frozen_model
from src.recall_blend import BLEND_CONFIGURATION
from src.tabpfn_study import threshold_summary


def test_strict_recall_excludes_exactly_eighty_percent():
    report = threshold_summary([1, 1, 1, 1, 0, 0, 1], [.95, .9, .85, .8, .75, .7, .65])
    assert report["highest_precision_with_recall_80"]["recall"] == 0.8
    strict = report["highest_precision_with_recall_above_80"]
    assert strict["recall"] == 1.0
    assert strict["precision"] == pytest.approx(5 / 7)
    assert strict["threshold"] == 0.65


def test_default_app_scores_the_adopted_blend(monkeypatch):
    monkeypatch.delenv("CHURN_FREEZE_MANIFEST", raising=False)
    bundle = load_frozen_model(APP_MANIFEST_PATH)
    point = bundle.decision["development_selection"]["operating_point"]
    assert bundle.decision["configuration"] == BLEND_CONFIGURATION
    assert point["recall"] > 0.8
    assert bundle.policy.threshold == point["threshold"]
    app = AppTest.from_file(str(PROJECT_ROOT / "src" / "app.py"), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert app.metric[0].label == "Estimated churn probability"
    assert any("freeze-v2" in caption.value for caption in app.caption)


def test_recall_release_requires_fresh_holdout(tmp_path):
    with pytest.raises(DataValidationError, match="fresh independent evaluation cohort"):
        evaluate_holdout(tmp_path / "holdout.csv", APP_MANIFEST_PATH, tmp_path / "metrics.json")
