# Telco Customer Churn

Predict customer churn from the IBM Telco Customer Churn dataset.

## Project structure

```
.
├── data/
│   ├── raw/            # original, untouched data (WA_Fn-UseC_-Telco-Customer-Churn.csv)
│   ├── interim/        # partially cleaned data
│   └── processed/      # final, model-ready data
├── notebooks/           # exploratory Jupyter notebooks, numbered (01_eda.ipynb, ...)
├── src/                 # reusable Python modules (data loading, features, models)
├── models/              # saved/serialized trained models
├── reports/
│   └── figures/          # exported charts for presentations
├── tests/                # unit tests for src/
├── requirements.txt
└── setup_ds_env.sh       # re-run anytime to rebuild/refresh the environment
```

## Setup

```bash
./setup_ds_env.sh telco-churn   # creates .venv, installs deps, registers Jupyter kernel
source .venv/bin/activate
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then select the **Python (telco-churn)** kernel in Jupyter/VS Code.
