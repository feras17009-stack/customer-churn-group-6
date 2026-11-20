# Customer Churn Prediction Using Machine Learning

**SIC capstone · Project Crew · Group 6 · November–December 2026**

Build a reproducible Python project that estimates the probability of churn for a customer in the IBM Telco Customer Churn dataset. Compare four classifiers, calibrate the selected model, and show its probability and decision in a Streamlit app. The dataset is a static snapshot. Describe results as performance on this sample and identify future forecasting accuracy and financial impact as open questions.

## Instructions for the implementing AI

Work from the root of `capstone-group-6/`. Read the existing README, dependency list, scripts, and notebooks. Follow the stages below, reuse completed work, and run each stage's checks before handing its outputs to the next owner. When assigned one stage, implement that stage and report its handoff. The implementing AI works with fixtures, the verified training release, and published aggregate results. Never read, load, or print the original labeled CSV, holdout file, or test labels. Implement intake and evaluation against fixtures and hand the exact production commands to Mohammed (backup: Majed). Continue with independent tasks while awaiting their outputs.

If a required input is missing, name its path and owner. Continue independent work, using a clearly marked fixture for code checks if helpful. Populate performance tables, plots, and report claims only from actual runs on the project data. At each handoff, record the files produced, exact reproduction command, check results, and any remaining dependency.

## Repository layout

| Path | Contents |
|---|---|
| `data/` | Training release manifest, split and fold assignments, and instructions for the separate custodian and developer workflows |
| `notebooks/` | Member experiments, exploratory analysis, and final tutorial |
| `src/` | Reusable preprocessing, training, comparison, calibration, evaluation, and app code |
| `models/` | Input schema, committed calibrated model, decision policy, and models/freeze_manifest.json |
| `results/` | Out-of-fold predictions, comparison tables, plots, and final test metrics |
| `reports/` | Final Word report, PowerPoint pitch, and Excel work breakdown structure (WBS) |
| `requirements.txt` | Dependencies and versions used in the verified run |
| `README.md` | Setup, data acquisition, reproduction steps, and app launch command |
| `tests/` | pytest checks for data isolation, pipelines, scoring, and app decisions |
| `.python-version`, `.gitignore` | Python 3.11.9; exclude virtual environments, caches, raw CSVs, local test partitions, and scratch runs. Track the frozen model and its metadata. |

## Delivery calendar and owners

All dates are in 2026, Saudi time. The team aims to submit on **Thursday, December 3**. The official deadline is **Friday, December 4 at 11:00 PM**.

| Gate | Work and handoff |
|---|---|
| Nov 20–21, Fri–Sat | Optional preparation only. Sunday’s handoff assumes no weekend work. |
| Nov 22, Sun | Mohammed releases the training asset. Majed ships a schema and stub factory. Nayar and Majed freeze the 19 input fields. Feras generates folds. Mohammed and Suliman record the primary cost assumptions. Smoke test one synthetic customer through a reloaded fixture model and the shared decision function. |
| Nov 23–24, Mon–Tue | Majed delivers the real pipeline. Feras and Jarallah run the eight configurations. Suliman develops calibration on fixtures in parallel. Nayar starts EDA, report and slides. Majed opens the tutorial and Mohammed opens the WBS. |
| Nov 25–26, Wed–Thu | Feras delivers model selection on Wednesday. Suliman calibrates the winner, sets the policy, and fits the final artifact. Feras verifies the manifest before Thursday’s freeze tag. If the freeze slips, move it to Sunday before evaluation. |
| Nov 29–30, Sun–Mon | Suliman and Mohammed integrate the app. Mohammed or Majed completes the evaluation and delivers aggregate test_metrics.json by Monday, Nov 30. |
| Dec 1, Tue | Jarallah independently reproduces training from the training asset and runs the tutorial in a fresh kernel. Its final test-results cell reads Monday’s aggregate JSON. If delayed, complete the training checks first and rerun the tutorial when the JSON arrives. |
| Dec 2–3, Wed–Thu | Nayar reconciles report and slides, Majed fixes tutorial issues, and Jarallah verifies any fixes. Test the presenting laptop and record the demo. Mohammed submits on Thursday, with Feras as backup. |
| Dec 4–5, Fri–Sat | Friday is an emergency submission buffer. Saturday is the scheduled pitch. |

## 1. Pin the dataset and isolate test labels

**Mohammed Altuwalah, with Majed Alsehli as backup.** Use IBM’s public 7,043-row, 21-column CSV. The reviewer verified the raw bytes and aggregate counts below. Mohammed repeats these checks in the isolated intake environment and records the measured SHA-256, counts, resolved IBM commit, permanent raw URL, attribution, and applicable redistribution terms in `data/README.md`. His recorded measurements are the project pin. [IBM source CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

| Intake check | Expected |
|---|---|
| Raw SHA-256 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |
| Header, dimensions, IDs | Original 21-column schema, 7,043 rows and 7,043 distinct nonempty customer IDs. Reject wider exports containing Churn Score, Churn Label, CLTV, or Churn Reason. |
| Target | 1,869 positive labels. Encode Yes as 1 and No as 0. |
| Charges | The 11 whitespace-only TotalCharges values coincide exactly with the 11 tenure-zero rows. All other charges parse as numbers. Leave raw predictor values in the release so training and inference use the same transformation. |

**Custodian workflow.** Mohammed or Majed runs `python -m src.split_data --source "$SOURCE_CSV" --private-dir "$PRIVATE_DIR" --release-dir "$RELEASE_DIR"` in an isolated environment. Trim customerID, reject empty IDs, remove exact duplicate records, and stop on conflicting rows for one ID. Preserve identical feature profiles belonging to distinct customers. Set row_id to the cleaned customerID, sort by row_id, and stratify 80/20 with seed 42. Keep the ID solely as the join key. Write 5,634 labeled rows to the release’s `train.csv` and 1,409 to private `holdout.csv`. Only the custodians retain the original CSV and holdout.

**Training handoff.** Mohammed creates an immutable, team-accessible training asset, subject to the recorded redistribution terms. If public redistribution is unclear, use the team’s authorized private project storage. Commit `data/train_manifest.json` with asset version, stable download location, byte count, row count, raw source digest, and training SHA-256. Commit `data/splits.csv` (row_id,split; no labels). Record the exact training download and checksum commands in the developer section of the README. Keep the full-source intake instructions in the custodian section.

**Developer workflow.** Feras, Jarallah, Suliman, Nayar, and Majed’s development environment start from that versioned training asset. Download to `data/train.csv`, verify the hash, and call `load_train()`. It checks exact training ID coverage and uniqueness against the split manifest and returns rows sorted by row_id. Every experiment and notebook uses it. On Dec 1 Jarallah follows this same path without acquiring the original CSV or rerunning intake.

## 2. Freeze a shared pipeline contract

**Majed owns preprocessing. Nayar Alsoulmi owns training-only EDA.** Export the following constructor from `src/preprocessing.py`. Return a fresh, unfitted, sklearn-cloneable Pipeline. Input is a DataFrame of the original 19 predictors. Obtain churn probability from the column of `predict_proba` corresponding to class 1.

```python
build_pipeline(model_name, *, weighting="none", seed=42) -> Pipeline
# model_name: logistic_regression | random_forest | xgboost | catboost
# weighting: none | balanced
```

Treat SeniorCitizen as a binary categorical predictor with allowed values 0 and 1. In both preprocessing routes normalize whitespace-only TotalCharges to missing. For a missing charge with tenure zero, return 0.0. For a missing charge at positive tenure, use the median fitted on that fit’s training rows. Preserve observed numeric charges. Document both branches in the input schema and let the app submit a missing charge to the pipeline. Parametrize pytest over both routes and both tenure cases, checking the transformed value and prediction after reload.

Derive the encoding route from the model name. For LR, RF, and XGBoost use one-hot encoding with unknown-category handling, scaling numeric features for LR. For CatBoost preserve a named DataFrame, stringify categorical values with an explicit missing token, and supply categorical column names through cat_features. Use pandas output on the CatBoost preprocessing path (set_output(transform="pandas")) and assert column names at the estimator boundary. Set allow_writing_files=False. For balanced LR/RF use class_weight="balanced"; for CatBoost use auto_class_weights="Balanced". Jarallah implements a cloneable XGBoost classifier wrapper that constructs its estimator inside fit and sets scale_pos_weight to the negative/positive ratio of the y passed to that fit (1 for unweighted). Preserve sklearn classifier semantics, classes_, and predict_proba. Test two fits with different prevalences, including instrumented inner calibration fits, and assert each receives its own ratio. Fit learned transforms and balanced weights inside each estimator fit, including calibration refits. Test cloning and prediction on a new category for both encoding routes. Persist names, types, allowed UI choices, and missing-value behavior in `models/input_schema.json`.

Nayar produces `notebooks/eda.ipynb` and `results/eda/` with prevalence, tenure, contract, charges, missingness, and subgroup denominators. Freeze the original feature list on Nov 22. Record feature-engineering ideas for future work.

## 3. Select a model by ranking performance

**Feras Mohammed Alharbi** owns Logistic Regression, Random Forest, and comparison. **Jarallah Al-Jarallah** owns XGBoost and CatBoost. Both use `src/cross_validate.py` with the shared factory. Use TabArena as the one-line reference for tabular model families. [TabArena](https://tabarena.ai/)

Feras writes `data/folds.csv` with `row_id,fold` from training rows sorted by row_id before applying five shuffled stratified training folds (seed 42). Freeze the eight configurations before scoring: four families, each unweighted and balanced. Use pinned-library defaults except for documented choices needed for convergence or reproducibility. Set both boosters to 200 iterations, CPU, one thread, seed 42, and disable early stopping. Pin XGBoost learning_rate=0.05 and max_depth=3. Keep other CatBoost learning parameters at pinned-version defaults and record resolved settings. Report comparisons as results at these fixed, untuned settings. Carry the exact winning settings into calibration and refit. This schedule assumes no hyperparameter search.

For each configuration write `results/oof/<configuration>.csv` with `row_id,fold,y_true,p_churn`. In `src/compare_models.py` report mean fold Average Precision (AP), ROC-AUC, and dispersion. Precision, Recall, and F1 at 0.5 are descriptive diagnostics for each score scale.

**Selection rule.** Find the best mean AP. Estimate its standard error as its five fold APs’ sample standard deviation divided by √5. Among configurations within that distance of the best, prefer LR, then RF, XGBoost, then CatBoost. Within a family prefer unweighted. Record paired fold AP differences, the eligibility cutoff, chosen configuration and rationale in `results/model_comparison.csv` and `results/selected_model.json`. The one-standard-error rule is a simplicity heuristic. Treat OOF summaries reused for selection and threshold choice as development estimates. Reserve the independent performance claim for the frozen holdout.

## 4. Calibrate probabilities and choose a contact policy

**Suliman AlTayar.** In `src/calibrate.py`, fit the selected baseline on each outer training fold. For the calibrated copy, use CalibratedClassifierCV with method="sigmoid", ensemble=False, and five shuffled stratified inner folds (seed 42) inside the outer training fold. The sigmoid uses inner out-of-fold scores and the single base estimator refits on the entire outer training fold. Use this same recipe on all training rows for the shipped artifact. Predict both on identical outer validation rows. Save paired Brier and AP differences by fold, a reliability plot with bin counts, and pooled OOF predictions under `results/calibration/`. Report any deterioration.

By Sunday, Nov 22, before viewing model or threshold results, Mohammed and Suliman record outreach cost per contacted customer (C), lost value per churner (L), and assumed probability of retaining a contacted would-be churner (s). For an executable illustrative default use C=1, L=10 and s=0.5 in common cost units. These are scenarios, not measured effects. For each threshold calculate:

```text
expected cost = C × (TP + FP) + L × [FN + (1 − s) × TP]
break-even probability = C / (s × L)
```

For the default, break-even is 0.20. With perfect retention it reduces to C/L. For zero retention and positive contact cost, contact none. Use analytical break-even as the default policy. To test a departure, hold out each of the five OOF fold blocks in turn. On the other four blocks, search unique calibrated probabilities plus explicit all/none policies, minimizing scenario cost and breaking ties by fewer contacts. On the reserved block compute the paired per-customer mean cost saving of that empirical policy versus break-even. Switch only when the mean of the five block savings exceeds its standard error (sample SD divided by √5). If it passes, select the deployable empirical threshold on all training OOF scores using the same rule. Otherwise retain break-even. Save the five paired savings, SE, decision, and policy type. This is a development stability check because the folds share training information. Repeat over s in {0.25, 0.5, 1} and C/L in {0.05, 0.10, 0.20}. Keep the predeclared primary scenario for deployment.

Report precision, recall, number contacted, and cost at the chosen operating point. From the OOF precision–recall curve, report maximum attainable precision among points with recall ≥80%, its actual recall, and whether any point also reaches 60% precision. These are exploratory targets. Assess them again at the frozen operating point on test. Handle zero-contact precision as undefined and label it explicitly.

## 5. Freeze and integrate the demo

Refit the selected pipeline with internal sigmoid calibration on all training data. Commit `models/churn_model.joblib`, `models/input_schema.json`, and `models/decision.json`. Include class mapping, policy, threshold, cost assumptions, configuration, and exact Python and library versions in the decision file. Save SHA-256 hashes of the model, decision file, and schema in `models/freeze_manifest.json` (paths and hashes; keep the manifest outside its own hash list). Verify the hashes, commit, and tag `freeze-v1`. Load the team's trusted artifact under the pinned environment.

Suliman builds `src/app.py`. Offer validated input choices from the schema. The underlying pipeline tolerates unseen categories on batch inputs. Share the decision function between the app and evaluator. Show the probability to one decimal percent and display the operating threshold or constant policy. Apply the threshold to full-precision values using ≥. If rounding makes the two displayed values equal, show additional precision or an explanation. Test below, at, and above the boundary.

## 6. Evaluate once and reproduce

Mohammed (backup: Majed) verifies the freeze manifest and runs the following command in the custodian environment. The AI tests the same interface on a synthetic holdout fixture.

```sh
python -m src.evaluate --holdout "$PRIVATE_DIR/holdout.csv" --manifest models/freeze_manifest.json --output results/test_metrics.json
```
 Save `results/test_metrics.json` with freeze ID, sample and positive counts, threshold/policy, confusion matrix, AP, ROC-AUC, Brier, Precision, Recall, F1, and scenario cost. Include total and per-customer costs for the frozen policy, contact-all, and contact-none under identical C, L, s. Report paired cost savings against each baseline. Within this single evaluation run, compute 95% percentile bootstrap intervals with 2,000 customer-level resamples, seed 42, using stored in-memory scores and the frozen policy. Reuse the same sampled indices for every metric and baseline. Include intervals for AP, ROC-AUC, Brier, precision, recall, F1, and per-customer costs and savings. Record valid replicate counts and use null where a statistic is undefined. State whether the point estimates reach the exploratory targets and show their uncertainty. Publish aggregate JSON only. Refuse to overwrite an existing evaluation for that freeze. After test exposure, log any correctness bug and affected outputs, preserve v1, review the fix, and tag a new version. Disclose repeat evaluation and prior exposure. Further model or threshold selection requires fresh independent data for a new independent performance claim.

Automate checks in `tests/` with pytest: checksum/schema and blank-charge rule at intake, row-order-invariant splits, full coverage with zero overlap, refusal of test IDs, identical fold membership and one OOF score per training row, cloneable pipelines, model reload and app parity, threshold boundary and constant policies, and confusion matrix totals. Test the evaluator on fixtures. On Nov 22 keep the smoke test to one fixture row through reload and the decision function. Exercise the full suite as components arrive and require it to pass before freeze. On Dec 1 Jarallah checks installation from pinned dependencies, training-only reproduction, saved artifact loading, clean-kernel tutorial, and `streamlit run src/app.py` on a fresh environment. Record commands and version or numeric differences in the README.

## 7. Package and submit

| Deliverable | Owner and acceptance check |
|---|---|
| Word report in `reports/` | Nayar integrates contributions. Trace each number to `results/`, distinguish OOF from test, and identify retention assumptions and snapshot limitations. |
| PowerPoint pitch in `reports/` | Nayar assembles slides with Suliman’s demo and recording. Include model choice, calibration, threshold trade-off, and holdout result. |
| Official Excel WBS in `reports/` | Mohammed records owners, dependencies, dates, and actual progress. Feras checks readiness as backup. |
| `notebooks/tutorial.ipynb` | Majed integrates the worked example. Jarallah runs it from a clean kernel without access to holdout labels. |

Mohammed confirms that all four deliverables open, the numbers match the frozen results, and the Slack and email submissions are recorded. Use Dec 4 only if the internal Dec 3 target slips.
