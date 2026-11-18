# ThyroAI — Thyroid Disorder Screening System

MCA Final Project | Machine Learning + Flask Web App

---

## Features

| Module | Description |
|---|---|
| Dataset | Real UCI thyroid disease dataset (7,674 patients after cleaning), diagnosis codes mapped to Normal / Hypothyroid / Hyperthyroid per the official UCI codebook |
| Features | Age, Sex, TSH, T3, T4 (real clinical; T4 = Total T4, UCI's "TT4" column relabeled to match standard lab-report naming) + Sleep, Stress, Exercise, Weight Gain (simulated lifestyle, probabilistically linked to diagnosis — see note below) |
| Diagnosis | **3-class**: Hyperthyroid / Hypothyroid / Normal |
| Risk Level | No / Medium / High Risk, derived internally from the model's prediction confidence (the confidence number itself is **not shown** to the user — only the resulting risk level) |
| ML Models | **4 models**: Logistic Regression, Decision Tree, Random Forest, XGBoost (Tuned) — SVM and KNN removed |
| Class Balancing | `class_weight='balanced'` (sklearn) / per-sample weights (XGBoost) — Hyperthyroid is the minority class (~3% of records) |
| Evaluation | Macro-averaged Accuracy, Precision, Recall, F1, ROC-AUC (one-vs-rest), 3×3 Confusion Matrix. (5-fold cross-validation step removed) |
| Explainability | SHAP summary plot + feature importance chart generated during training (saved to `static/`, not shown on the result page per project requirements) |
| User Authentication | Register / Login / Logout — every route requires a logged-in session |
| Flask App | Input form (incl. patient name) → diagnosis + risk level (no confidence/probability numbers shown) |
| Risk Analytics Dashboard | Per-user stats: total predictions, High/Medium/No risk counts, avg TSH/sleep, risk distribution pie + **TSH trend line** with a shaded normal-range band and risk-level-colored points |
| PDF Reports | Lab-report-style layout (Test Description / Value(s) / Reference Range) — patient name, diagnosis, risk level. No probability tables, no confidence shown |
| Database | SQLite stores users and predictions (each prediction linked to a user, includes patient name) |
| Theme | Clean white/clinical color scheme (teal/cyan accents) — no dark mode |

---

## Setup

```bash
pip install -r requirements.txt
python generate_and_train.py    # trains all models, saves to model/
python app.py                   # starts Flask on http://127.0.0.1:5000
```

> **Note:** requires `xgboost` (`pip install xgboost`). Falls back to GradientBoostingClassifier if unavailable.

> **Auth note:** Sessions are signed with `THYROAI_SECRET_KEY`. Set this environment variable
> to a fixed random value in production — otherwise a new key is generated on every restart
> and all users get logged out each time the server restarts.
> ```bash
> export THYROAI_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
> ```

---

## Input Features

| Category | Feature | Notes |
|---|---|---|
| Demographic | Age, Sex | — |
| Clinical | TSH (mIU/L) | Normal: 0.5–4.5. Strongest predictor — low TSH suggests hyperthyroid, high TSH suggests hypothyroid |
| Clinical | T3 (nmol/L) | Normal: 1.1–2.5 |
| Clinical | T4 (nmol/L) | Normal: 60–150 |
| Symptom | Weight Gain | Yes/No |
| Lifestyle | Sleep Hours | Hrs/night |
| Lifestyle | Stress Level | Low/Medium/High |
| Lifestyle | Exercise Frequency | Never/Weekly/Daily |

---

## Important methodology note — diagnosis labels vs. lifestyle features

The **diagnosis label** (Hyperthyroid / Hypothyroid / Normal) comes from the **real** UCI
diagnostic codes (`target` column in `thyroidDF.csv`), mapped using the standard codebook:
- `-` → Normal
- Codes starting with A–D → Hyperthyroid family
- Codes starting with F–H → Hypothyroid family

This is genuine clinical ground truth — not synthetic.

The **lifestyle features** (sleep, stress, exercise, weight gain) are **simulated**, with their
probability distributions skewed based on the diagnosis (e.g. abnormal diagnoses get a higher
probability of poor sleep). This is a known limitation: it means part of the model's measured
accuracy comes from this designed correlation, not from genuine lifestyle signal. Measured impact:
training on clinical features alone (age, sex, TSH, T3, T4) already achieves ~90% accuracy /
~95% AUC on the binary detection task; the simulated lifestyle features add a few points on top.
Be ready to explain this distinction if asked.

---

## Risk level logic

Risk level is computed internally from the model's confidence in its own predicted class
(see `confidence_to_risk_level()` in `app.py`). **The confidence number itself is not shown
anywhere in the UI or PDF report** — only the resulting risk level (No/Medium/High) is displayed:

- Predicted **Normal**: risk is "No Risk" unless confidence is unusually low (<60%), in which
  case it's flagged "Medium Risk" since the model itself is uncertain.
- Predicted **Hypothyroid / Hyperthyroid**:
  - confidence ≥ 80% → **High Risk**
  - 55% ≤ confidence < 80% → **Medium Risk**
  - confidence < 55% → **No Risk** (model isn't confident enough to flag concern)

---

## Disclaimer

This is a **screening tool only** and does NOT constitute medical advice.
Always consult a qualified healthcare professional for diagnosis and treatment.
