# Customer Churn Prediction Using Machine Learning

SIC capstone · Project Crew · Group 6

[Open the Streamlit app](https://sic-group6-customer-churn.streamlit.app/)

The project estimates churn probability on the IBM Telco Customer Churn snapshot. Its scores describe performance on this sample. Future forecasting accuracy and financial impact remain open questions.

## Adopted outreach model

Streamlit uses **freeze-v2**, a log-odds blend of **75% Random Forest** and **25% CatBoost**. Across 406 development comparisons, it achieved the highest precision among points with recall **strictly greater than 80%**. At the frozen threshold of **27.4381%**, the five IBM training folds gave **53.60% precision**, **80.13% recall** and **0.6424 F1**. The threshold applies to the full score. These are development estimates from the folds used for model and threshold selection. Independent validation requires a fresh cohort.

[Full training-fold comparison](results/joint-target-tuning.json) · [Adopted decision](models/freeze-v2/decision.json) · [Original XGBoost holdout evaluation](results/test_metrics.json)

The original holdout figures, study report, pitch and tutorial document **freeze-v1**. TabPFN-3.5 is available for local research using a separately obtained checkpoint. [TabPFN-3.5 license](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE)

## Setup and fixture checks

Install Python **3.11.9** and run these commands from the project root. [Python 3.11.9 downloads](https://www.python.org/downloads/release/python-3119/)

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q tests
```

On macOS or Linux, use `python3.11 -m venv .venv` and `.venv/bin/python` for the remaining commands. Confirm that the interpreter reports `Python 3.11.9`. The tests generate fictional records in temporary directories.

## Developer workflow

The IBM deployment uses a local-only release (`local_only: true`). Recreate its exact bytes in a separate staging directory using the pinned source and the [custodian intake](data/README.md#custodian-intake). Copy the resulting `train.csv` to `data/train.csv` and run `python -m src.train_data verify`. Keep per-customer development scores in the local workspace. Recreate them using the training comparison and calibration commands before running the independent reproduction checks.

For a hosted release that declares a `download_url`, use the reviewed manifest and the access method for the project's authorized storage. These PowerShell commands cover a direct or session-authorized HTTPS download.

```powershell
$manifest = Get-Content -LiteralPath data/train_manifest.json -Raw | ConvertFrom-Json
Invoke-WebRequest -Uri $manifest.download_url -OutFile data/train.csv
$actual = (Get-FileHash -LiteralPath data/train.csv -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $manifest.train_sha256) { throw 'Training file SHA-256 differs from the manifest.' }
.\.venv\Scripts\python.exe -m src.train_data verify
```

The commands from this point assume the virtual environment is active. Activate it in PowerShell with `.\.venv\Scripts\Activate.ps1` or use the virtual environment's Python directly as above. For authenticated storage, use the project storage session or its documented download client to save those same bytes at `data/train.csv`.

Load data through `from src.train_data import load_train`, then `train = load_train()`. The loader checks the byte count, SHA-256, schema, and exact training membership and returns rows sorted by `row_id`. Pass the columns in `src.data_contract.FEATURE_COLUMNS` to estimators and use `Churn` as the binary target. Use `row_id` for joins.

Run `python -m src.eda` to write the training-only counts, subgroup rates, charges, missingness, and figure to `results/eda/`. The same summaries are in `notebooks/eda.ipynb`. The input contract is defined in `models/input_schema.json`. The illustrative primary cost assumptions were recorded in `data/cost_scenario.json` before scoring.

### Training comparison

```sh
python -m src.cross_validate --write-folds
python -m src.cross_validate
python -m src.compare_models
```

Commit the verified `data/folds.csv`. Its five shuffled stratified fold IDs are numbered 0–4 and use seed 42. Omit `--write-folds` when reproducing an already committed assignment. `--configuration` may be repeated to split scoring across teammates, for example `--configuration random_forest_none --configuration random_forest_balanced`. For a fresh run against the same assignments, pass a new `--results-dir` to both scoring and comparison.

The eight configurations are fixed before scoring in `src/cross_validate.py`. Logistic regression uses standard scaling and `max_iter=1000` for convergence. Random forest and both boosters use one thread and seed 42. Both boosters use 200 iterations on the CPU with early stopping disabled. XGBoost sets learning rate 0.05 and maximum depth 3. CatBoost uses the pinned version's other defaults, including any values the library resolves per fit. The fold settings are written alongside each OOF file. On Windows and Linux x86-64 use the pinned CPU-only XGBoost distribution, with the same version of `xgboost` on the other supported platforms. [TabArena tabular model families](https://tabarena.ai/)

`results/model_comparison.csv` records fold AP and ROC-AUC, sample standard deviations, paired AP differences from the best mean AP, and pooled diagnostics at 0.5. Blank precision means the configuration contacted no customers at that cutoff. `results/selected_model.json` records eligibility and the choice by the plan's simplicity heuristic. These comparisons describe fixed, untuned settings on training folds.

### Calibration and outreach policy

Run `python -m src.calibrate` with the same `--folds` and `--results-dir` used for the comparison. The command fits a baseline and a sigmoid-calibrated copy on each outer training block. Inner stratified folds fit the sigmoid and the base estimator then refits on the entire outer training block. See `results/calibration/paired_metrics.csv` for paired Brier and AP, `reliability.png` and `reliability.csv` for reliability and bin counts, and `report.json` and `sensitivity.csv` for the policy and cost trade-offs.

The primary assumptions are illustrative: one cost unit per contact, ten lost per churner, and 50% retention among contacted would-be churners. The analytical break-even probability is 20%. The policy switches to an empirical cutoff only when held-out OOF block savings exceed their standard error. OOF performance and the savings check are development estimates. Retention and scenario costs describe assumptions, and the snapshot cannot establish actual impact.

### Model freeze and demo

```sh
python -m pytest -q tests
python -m src.freeze_model
python -m src.freeze_model --verify
streamlit run src/app.py
```

Refit after the training-only comparison and calibration are reviewed. Commit `models/churn_model.joblib`, `models/input_schema.json`, `models/decision.json`, and `models/freeze_manifest.json` with the reviewed code, release metadata, folds and development results. Tag that commit `freeze-v1` before evaluation. The verifier hashes the same bytes it loads and checks the recorded runtime. Use the team's trusted manifest and artifacts. The app uses the model's schema and applies the policy to the unrounded score.

For Streamlit Community Cloud, deploy `src/app.py` from `main` and select Python 3.11 in Advanced settings. The server loads the verified model from `models/` and requires the recorded library versions. The app accepts Python patch releases within the version it was trained on. The freeze verifier, evaluator and reproduction commands continue to require the exact recorded Python version.

## Custodian workflow

Mohammed Altuwalah or Majed Alsehli uses the separate intake environment and follows the [production intake and release commands](data/README.md#custodian-intake). The custodian publishes the training asset and its verified metadata after recording source provenance and applicable redistribution terms.

### Reproduce the adopted strict-recall release

Set up the verified local training release and regenerate the development scores. The experimental TabPFN commands use a separate environment with the optional `requirements-tabpfn.txt` package and an authorized checkpoint at `models/tabpfn/tabpfn-v3.5-20260909.safetensors`. Keep checkpoints and per-customer scores in local storage. The adopted blend refit uses the ordinary project dependencies.

```powershell
python -m src.tune_classical
python -m src.tune_operating_point
python -m src.tune_tabpfn --config raw3
python -m src.tune_tabpfn --config derived1
python -m src.tuning_report --tabpfn raw3 derived1 --extra --output output/joint-target-tuning.json
python -m src.recall_release --report output/joint-target-tuning.json --model-dir .scratch/refit-freeze-v2
python -m src.freeze_model --verify --model-dir .scratch/refit-freeze-v2
```

To reproduce the adopted blend and threshold directly from the committed aggregate report, run `python -m src.recall_release --model-dir .scratch/refit-freeze-v2`. The refit recomputes all its out-of-fold predictions and confirms the selected operating point before saving the estimator. The evaluated IBM holdout belongs to the original release.

### One-time holdout evaluation for freeze-v1

Work from the reviewed `freeze-v1` revision in the isolated environment with the same pinned dependencies. Keep `PRIVATE_DIR` pointed at the intake's restricted directory.

```sh
python -m src.freeze_model --verify
python -m src.evaluate --holdout "$PRIVATE_DIR/holdout.csv" --manifest models/freeze_manifest.json --output results/test_metrics.json
```

In PowerShell, use `--holdout (Join-Path $env:PRIVATE_DIR 'holdout.csv')` with the same manifest and output arguments. Publish only `results/test_metrics.json` from evaluation. The report uses 2,000 paired customer-level percentile bootstrap resamples with seed 42. Undefined values are JSON `null`, with valid replicate counts alongside intervals.

The evaluator creates a per-freeze audit file in `PRIVATE_DIR/.churn-evaluations/` before reading labels and refuses repeated evaluation in that environment. If a bug is found after exposure, preserve the first freeze, audit and published outputs. Log the error and affected metrics, review the correction, and use a new freeze tag with repeat-evaluation disclosure. Further selection requires new independent data for a new independent performance claim.

## Independent reproduction and handoff

Jarallah runs these commands in a fresh Python 3.11.9 environment from the reviewed project revision after acquiring the verified training release. The reproduction command reruns the eight configurations, verifies calibration and the decisions, compares a fresh refit with the trusted frozen model, and runs the EDA and tutorial in separate clean kernels.

```sh
python -m pip check
python -m pytest -q tests
python -m src.reproduce
streamlit run src/app.py
```

The default output is `.scratch/reproduction/`. Record versions and observed differences from `training_reproduction.json` in the reproduction handoff. The probability comparison uses relative tolerance `1e-7` and absolute tolerance `1e-8`. The last tutorial cell reads `results/test_metrics.json` only. If the custodian has yet to publish it, run the notebooks again afterward with `python -m src.reproduce --notebooks-only --output-dir .scratch/reproduction-with-evaluation`.

## Phase verification

Run these commands from the root with the pinned virtual environment active. Each test uses synthetic data in temporary directories.

All 66 pytest checks passed on Windows x86-64 with Python 3.11.9 and the versions in `requirements.txt`. The tests cover the default serving bundle, the strict recall cutoff, the old holdout guard, and validation of dependent phone and internet fields in Streamlit. The original independent fixture run had a maximum absolute score difference of 0. The EDA and tutorial ran in fresh kernels. The tutorial passed both before and after publication of the aggregate fixture results.

| Phase | Delivered | Fixture check |
|---|---|---|
| 1. Setup and isolation | Python pins, custodian intake, training manifest and loader | `python -m pytest -q tests/test_data_release.py` |
| 2. Shared pipeline | Input schema, four estimators, two encoding paths and EDA notebook | `python -m pytest -q tests/test_preprocessing.py tests/test_eda.py` |
| 3. Comparison | Deterministic folds, eight OOF runs, one-standard-error selection | `python -m pytest -q tests/test_model_selection.py` |
| 4. Calibration and policy | Nested sigmoid calibration, reliability, contact decisions and sensitivity | `python -m pytest -q tests/test_calibration.py` |
| 5. Freeze and demo | Hash-verified calibrated bundle and Streamlit form | `python -m pytest -q tests/test_freeze_and_app.py` |
| 6. Evaluation and reproduction | One-time aggregate evaluation and independent training checks | `python -m pytest -q tests/test_evaluation.py` |
| 7. Packaging | Worked tutorial, Word study methods, PowerPoint pitch and Excel WBS | `python -m pytest -q tests` and the clean-kernel and release checks above |

## Delivery files

```text
.gitignore
.python-version
README.md
requirements.txt
data/
  README.md
  cost_scenario.json
models/
  input_schema.json
notebooks/
  eda.ipynb
  tutorial.ipynb
reports/
  churn_project_pitch.pptx
  churn_study_methods.docx
  group6_work_breakdown.xlsx
src/
  __init__.py
  app.py
  artifacts.py
  calibrate.py
  compare_models.py
  cross_validate.py
  data_contract.py
  eda.py
  evaluate.py
  freeze_model.py
  input_schema.py
  metrics.py
  policy.py
  preprocessing.py
  reproduce.py
  split_data.py
  train_data.py
tests/
  conftest.py
  test_calibration.py
  test_data_release.py
  test_eda.py
  test_evaluation.py
  test_freeze_and_app.py
  test_model_selection.py
  test_preprocessing.py
```

The verified production handoffs create the additional files below.

| Owner | Files and acceptance check |
|---|---|
| Mohammed Altuwalah, backup Majed Alsehli | Add the measured source record to `data/README.md`, commit `data/train_manifest.json` and `data/splits.csv`, and publish the versioned asset downloaded as `data/train.csv`. Verify it with `python -m src.train_data verify`. |
| Feras Mohammed Alharbi and Jarallah Al-Jarallah | Generate `data/folds.csv` and `results/oof/` from that release, then publish `results/model_comparison.csv`, `results/model_comparison.png` and `results/selected_model.json`. |
| Suliman AlTayar and Feras Mohammed Alharbi | Generate `results/calibration/`, review the model and policy, verify `models/churn_model.joblib`, `models/decision.json` and `models/freeze_manifest.json`, then commit and tag `freeze-v1`. |
| Mohammed Altuwalah, Nayar Alsoulmi, backup Majed Alsehli | Run the custodian command once and publish `results/test_metrics.json`. |
| Jarallah Al-Jarallah, Nayar Alsoulmi and Majed Alsehli | Reproduce the training, generate `results/eda/`, reconcile the report and pitch with the empirical results, and rerun the tutorial. |
| Suliman AlTayar and Mohammed Altuwalah | Verify the app on the presenting laptop, record the production demo, then confirm the four deliverables and submit. |
