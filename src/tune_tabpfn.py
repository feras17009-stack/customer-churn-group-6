"""Run additional TabPFN-3.5 configurations on the verified training folds."""

import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from src.cross_validate import N_FOLDS
from src.data_contract import PROJECT_ROOT, SPLIT_SEED, DataValidationError
from src.preprocessing import CATEGORICAL_COLUMNS, churn_probabilities
from src.tabpfn_study import CHECKPOINT, CHECKPOINT_SHA256, sha256_of, threshold_summary
from src.tuning_data import alternatives, read_score, save_score, training_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("raw3", "derived1"), required=True)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TABPFN_NO_BROWSER"] = "1"
    import torch
    from tabpfn import TabPFNClassifier

    torch.set_num_threads(args.threads)
    checkpoint = PROJECT_ROOT / "models" / "tabpfn" / CHECKPOINT
    if sha256_of(checkpoint) != CHECKPOINT_SHA256:
        raise DataValidationError("The TabPFN checkpoint does not match its official digest.")

    _, _, raw, labels, membership, context = training_inputs()
    derived = args.config == "derived1"
    x = alternatives(raw) if derived else raw
    n_estimators = 1 if derived else 3
    categorical = [x.columns.get_loc(name) for name in CATEGORICAL_COLUMNS]
    name = "tabpfn_" + args.config
    probabilities = np.empty(len(labels), dtype=float)
    for fold in range(N_FOLDS):
        valid = membership == fold
        scores = read_score(name, fold, int(valid.sum()), context)
        if scores is None:
            print(f"{name}: scoring fold {fold + 1} of {N_FOLDS}.", flush=True)
            start = perf_counter()
            model = TabPFNClassifier(
                model_path=Path(checkpoint), device="cpu", n_estimators=n_estimators, random_state=SPLIT_SEED,
                memory_saving_mode="auto", categorical_features_indices=categorical,
            )
            model.fit(x.loc[~valid], labels[~valid])
            scores = churn_probabilities(model, x.loc[valid])
            save_score(name, fold, scores, context)
            print(f"{name}: fold {fold + 1} completed in {perf_counter() - start:.1f}s.", flush=True)
        probabilities[valid] = scores
    point = threshold_summary(labels, probabilities)["highest_precision_with_recall_80"]
    print(f"{name}: best precision at recall of at least 80% was {point['precision']:.2%}.", flush=True)


if __name__ == "__main__":
    main()
