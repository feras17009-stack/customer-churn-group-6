# customer-churn-group-6
Project Crew Group 6 capstone for customer churn prediction and model calibration.


## Project layout

| Path | Purpose |
| --- | --- |
| [`data/`](data/) | Dataset and acquisition instructions |
| [`notebooks/`](notebooks/) | Team experiments and tutorial notebooks |
| [`src/`](src/) | Preprocessing, training, calibration, and evaluation code |
| [`models/`](models/) | Small model files or links to stored models |
| [`results/`](results/) | Tables, figures, and evaluation metrics |
| [`reports/`](reports/) | Final report and presentation |
| [`requirements.txt`](requirements.txt) | Python dependencies |

## Setup

Create and activate a Python virtual environment, then install dependencies once the team has populated `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Follow the [dataset instructions](data/README.md) to obtain the CSV. Training, evaluation, and demo commands will be documented here as their entry points are added.
