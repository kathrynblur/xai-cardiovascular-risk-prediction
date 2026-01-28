# LIME directory — how to use

This folder contains everything needed to run LIME explanations and view results.

## What you need
To make the LIME pipeline work you need **two files** from this folder:
- `lime_explain.py` — the CLI script that actually runs LIME.
- `lime_full_report.ipynb` — the notebook that runs the script and shows results.

If you copy this folder elsewhere, **at minimum copy those two files** and keep the same project structure (`models/`, `processed/`).

---

## How to run (CLI)
Run from the project root:

### Local explanation (one patient)
```
python3 explainability/lime/lime_explain.py --index 0
```
Outputs:
- `explainability/lime/lime_explanation.html`
- `explainability/lime/lime_explanation.txt`

### Global explanation (many patients)
```
python3 explainability/lime/lime_explain.py --aggregate --max-instances 1000 --progress-every 100
```
Outputs:
- `explainability/lime/lime_global_summary.csv`
- `explainability/lime/lime_global_summary.png`

---

## How to run (Notebook)
Open **`lime_full_report.ipynb`** and click **Run All**.
The notebook will:
1) run local LIME
2) show HTML + TXT output inline
3) run global LIME
4) show the table + plot inline

---

## What you can change to get different results
### In the **CLI** (lime_explain.py)
- `--index` — choose another patient (local explanation)
- `--num-features` — number of top features per explanation (default 10)
- `--num-samples` — LIME samples per explanation (default 5000)
- `--aggregate` — switch to global aggregation mode
- `--max-instances` — limit number of patients (0 = all)
- `--progress-every` — progress log frequency

### In the **notebook** (lime_full_report.ipynb)
You can edit the command cells:
- change `--index` for local case
- change `--max-instances` for global aggregation
- change `--num-features` / `--num-samples`

---

## File outputs (you can move them here)
If you want all results in this folder, copy the outputs here after running:
- `lime_explanation.html`
- `lime_explanation.txt`
- `lime_global_summary.csv`
- `lime_global_summary.png`

---

## Troubleshooting
- **FileNotFoundError** → run from project root or check that `models/outputs/` and `processed/` exist.
- **No module named catboost** → install it: `python3 -m pip install catboost`
- **Non‑numeric features error** → don’t disable preprocessing; the script auto‑applies it.
