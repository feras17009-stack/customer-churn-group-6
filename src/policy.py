"""Cost assumptions, full-precision contact decisions and policy selection."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import read_json
from src.cross_validate import N_FOLDS
from src.data_contract import DATA_DIR, DataValidationError


COST_SCENARIO_PATH = DATA_DIR / "cost_scenario.json"


@dataclass(frozen=True)
class ContactPolicy:
    policy_type: str
    threshold: float | None = None

    def __post_init__(self):
        if self.policy_type == "threshold":
            if type(self.threshold) not in (float, int) or not 0 <= self.threshold <= 1:
                raise DataValidationError("A threshold policy requires a finite threshold in [0, 1].")
        elif self.policy_type not in ("contact_all", "contact_none") or self.threshold is not None:
            raise DataValidationError("A constant policy must be contact_all or contact_none with a null threshold.")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, entry) -> "ContactPolicy":
        if not isinstance(entry, dict) or set(entry) != {"policy_type", "threshold"}:
            raise DataValidationError("Invalid contact policy record.")
        return cls(entry["policy_type"], entry["threshold"])


@dataclass(frozen=True)
class CostScenario:
    contact_cost: float
    lost_value: float
    retention_probability: float

    def __post_init__(self):
        for value in (self.contact_cost, self.lost_value, self.retention_probability):
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise DataValidationError("Scenario inputs must be finite nonnegative numbers.")
        if self.retention_probability > 1:
            raise DataValidationError("Retention probability cannot exceed 1.")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, entry) -> "CostScenario":
        fields = ("contact_cost", "lost_value", "retention_probability")
        if not isinstance(entry, dict) or not set(fields).issubset(entry):
            raise DataValidationError("Scenario requires contact_cost, lost_value and retention_probability.")
        return cls(*(entry[field] for field in fields))

    def break_even_policy(self) -> ContactPolicy:
        benefit = self.lost_value * self.retention_probability
        if benefit == 0 or self.contact_cost > benefit:
            return ContactPolicy("contact_none")
        return ContactPolicy("threshold", self.contact_cost / benefit)


def load_cost_scenario(path: Path = COST_SCENARIO_PATH) -> CostScenario:
    return CostScenario.from_dict(read_json(path))


def probability_array(probabilities) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise DataValidationError("Churn probabilities must be a one-dimensional array in [0, 1].")
    return values


def contact_decisions(probabilities, policy: ContactPolicy) -> np.ndarray:
    values = probability_array(probabilities)
    if policy.policy_type == "threshold":
        return values >= policy.threshold
    return np.full(values.shape, policy.policy_type == "contact_all", dtype=bool)


def customer_costs(labels, contacts, scenario: CostScenario) -> np.ndarray:
    y = np.asarray(labels)
    contacted = np.asarray(contacts, dtype=bool)
    if y.ndim != 1 or contacted.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise DataValidationError("Scenario costs require matching contact decisions and binary labels.")
    return scenario.contact_cost * contacted + scenario.lost_value * y * (1 - scenario.retention_probability * contacted)


def minimum_cost_policy(labels, probabilities, scenario: CostScenario) -> ContactPolicy:
    scores = probability_array(probabilities)
    y = np.asarray(labels)
    if y.shape != scores.shape or y.size == 0 or not np.isin(y, [0, 1]).all():
        raise DataValidationError("Policy search requires matching nonempty probabilities and binary labels.")
    order = np.argsort(-scores, kind="stable")
    descending = scores[order]
    ends = np.r_[0, np.flatnonzero(descending[:-1] != descending[1:]) + 1, len(scores)]
    true_positives = np.r_[0, np.cumsum(y[order])][ends]
    incremental_cost = scenario.contact_cost * ends - (scenario.retention_probability * scenario.lost_value) * true_positives
    # Adjacent candidates can differ only by the rounding of the same cost expression.
    tied = np.isclose(incremental_cost, incremental_cost.min(), rtol=0, atol=np.spacing(np.abs(incremental_cost).max()) * 4)
    contacts = int(ends[np.flatnonzero(tied)[0]])
    if contacts == 0:
        return ContactPolicy("contact_none")
    if contacts == len(scores):
        return ContactPolicy("contact_all")
    return ContactPolicy("threshold", float(descending[contacts - 1]))


def select_policy(oof: pd.DataFrame, scenario: CostScenario) -> dict:
    if set(oof["fold"]) != set(range(N_FOLDS)):
        raise DataValidationError("Policy checking requires all five training OOF blocks.")
    y = oof["y_true"].to_numpy()
    scores = probability_array(oof["p_churn"])
    fold_ids = oof["fold"].to_numpy()
    break_even = scenario.break_even_policy()
    checks = []
    for fold in range(N_FOLDS):
        reserved = fold_ids == fold
        candidate = minimum_cost_policy(y[~reserved], scores[~reserved], scenario)
        empirical_cost = customer_costs(y[reserved], contact_decisions(scores[reserved], candidate), scenario)
        analytical_cost = customer_costs(y[reserved], contact_decisions(scores[reserved], break_even), scenario)
        checks.append({
            "fold": fold, "customers": int(reserved.sum()), "candidate": candidate.as_dict(),
            "per_customer_saving": float((analytical_cost - empirical_cost).mean()),
        })
    savings = np.asarray([item["per_customer_saving"] for item in checks])
    mean, se = float(savings.mean()), float(savings.std(ddof=1) / np.sqrt(N_FOLDS))
    use_empirical = mean > se
    chosen = minimum_cost_policy(y, scores, scenario) if use_empirical else break_even
    return {
        "method": "empirical" if use_empirical else "break_even", "policy": chosen.as_dict(),
        "break_even_policy": break_even.as_dict(), "paired_savings": checks,
        "mean_per_customer_saving": mean, "standard_error": se,
    }
