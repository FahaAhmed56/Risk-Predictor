# 🎓 Student Support AI — ML Portfolio Project

[![Python CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](#testing)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A GitHub-ready **end-to-end machine-learning project** that estimates an **illustrative** student support risk score from attendance, study time, prior scores, assignment completion, and sleep hours. Includes a Streamlit interface, synthetic dataset, training, model evaluation, tests, and CI.

> **Important:** The data and labels are entirely *synthetic*, generated from an invented mathematical rule. Results do not validate any claim about real students. Never use this model for educational decisions or to penalize individuals.

## Screens and pipeline

```text
Synthetic dataset generator (src/data.py)
    -> data/synthetic_students.csv
    -> Train/test split and random-forest model (src/train.py)
    -> models/risk_model.joblib + models/metrics.json
    -> Streamlit dashboard (app/streamlit_app.py)
    -> Optional CLI prediction (src/predict.py)
```

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.data
python -m src.train
streamlit run app/streamlit_app.py
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe -m pip install -r requirements.txt` and run commands using `.venv\Scripts\python.exe` directly.

## Quick start (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data
python -m src.train
streamlit run app/streamlit_app.py
```

The app usually opens at **http://localhost:8501**.

A small pre-generated example CSV and trained model are **included** in this repository. You can run the UI directly after installing requirements, or run the generator and trainer to reproduce them.

## Example prediction from terminal

```bash
python -m src.predict --json '{"attendance_pct":80,"study_hours_week":12,"previous_score":75,"assignments_completed_pct":90,"sleep_hours":7}'
```

## Project files

| Path | Purpose |
| --- | --- |
| `src/data.py` | Reproducible synthetic data generation |
| `src/features.py` | Shared feature schema and validation |
| `src/train.py` | Random forest, held-out evaluation, saved model |
| `src/predict.py` | Inference function and command-line interface |
| `app/streamlit_app.py` | Interactive website |
| `data/synthetic_students.csv` | Generated demonstration dataset |
| `models/risk_model.joblib` | Pre-trained example model |
| `models/metrics.json` | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix and feature importances |
| `tests/test_project.py` | Input, determinism and integration tests |
| `.github/workflows/ci.yml` | Automated pytest on GitHub push/PR |

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

## ML approach

1. Generate reproducible synthetic observations, with noisy outcomes.
2. Split into 80% training and 20% test examples (stratified).
3. Apply median imputation, then a class-balanced random forest.
4. Evaluate on the held-out test subset using accuracy, precision, recall, F1, ROC-AUC and a confusion matrix.
5. Save the pipeline to `joblib` and use predicted probabilities for UI display.

**Limitations:** Synthetic labels encode the generator's assumptions; correlations/feature importance do not prove causation. The 0.5 display threshold is arbitrary; the probabilities are not calibrated against actual outcomes. There is no real-world validation or fairness assessment. Avoid real-student data without permission and appropriate privacy safeguards.

## Push to GitHub

Create an **empty** GitHub repository named `student-support-ai`, then open a terminal in this folder:

```bash
git init
git add .
git commit -m "Initial commit: student support ML demo"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/student-support-ai.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Do not upload `.venv`, API keys or private data.

## Optional deployment

On Streamlit Community Cloud, connect the repository and specify `app/streamlit_app.py` as the main app file. The sample trained model is included so the app can load without retraining. Install/run only artifacts from trusted repositories since `joblib` uses Python pickle.

## Potential future upgrades

Use a consented, ethically reviewed dataset; add cross-validation and probability calibration; explore model drift and subgroup fairness; add explanation methods and model versioning. This demo is a portfolio starter, not a production decision system.

## License

MIT (see `LICENSE`).
