"""Shared churn ranking and operating-point metrics."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def operating_metrics(labels, contacts) -> dict:
    y = np.asarray(labels, dtype=bool)
    contacted = np.asarray(contacts, dtype=bool)
    tp = int((y & contacted).sum())
    fp = int((~y & contacted).sum())
    tn = int((~y & ~contacted).sum())
    fn = int((y & ~contacted).sum())
    total = 2 * tp + fp + fn
    return {
        "customers": int(y.size), "positives": tp + fn, "contacted": tp + fp,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / total if total else None,
    }


def probability_metrics(labels, probabilities) -> dict:
    y = np.asarray(labels)
    scores = np.asarray(probabilities)
    positives = y.sum()
    return {
        "average_precision": float(average_precision_score(y, scores)) if positives else None,
        "roc_auc": float(roc_auc_score(y, scores)) if 0 < positives < len(y) else None,
        "brier": float(np.square(scores - y).mean()),
    }


def precision_recall_target(labels, probabilities) -> dict:
    if not np.any(labels):
        return {"max_precision": None, "recall_at_max_precision": None, "reaches_both_targets": None}
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    eligible = recall >= 0.8
    value, actual_recall = max(zip(precision[eligible], recall[eligible], strict=True))
    return {"max_precision": float(value), "recall_at_max_precision": float(actual_recall), "reaches_both_targets": bool(value >= 0.6)}
