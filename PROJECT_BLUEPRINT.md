# Early Fraud Signal System for Bank Account Openings
## IS557 Applied ML — UIUC | Rahul, Anisha, Ananyaa

---

## PROJECT GOAL
Predict fraudulent bank account applications at submission time
before the account is approved or any money moves.
Research question: Can application-time signals alone predict
fraudulent bank account openings before approval?

---

## DATASET
- File: main/Base.csv
- Source: BAF Base Dataset — NeurIPS 2022 by Feedzai
- Size: 1,000,000 records, 32 features
- Target: fraud_bool (0 = legitimate, 1 = fraud)
- Fraud Rate: 1.1% (11,029 fraud / 988,971 legitimate)
- Temporal: month column 0–7
- Train: months 0–5 | Test: months 6–7 (NO random split)

---

## CLASS IMBALANCE
- Ratio: 988,971 vs 11,029 = 89.7x imbalance
- scale_pos_weight = 89.7 for XGBoost
- SMOTE on training set only — never on test set
- NEVER use accuracy as metric

---

## FEATURES (32 total)
### Applicant Profile
income, customer_age, credit_risk_score, employment_status,
proposed_credit_limit

### Velocity Signals
velocity_6h, velocity_24h, velocity_4w, zip_count_4w,
bank_branch_count_8w

### Device & Digital
device_fraud_count, device_os, email_is_free, keep_alive_session,
device_distinct_emails_8w, name_email_similarity, source

### Banking History
bank_months_count, has_other_cards, phone_mobile_valid,
phone_home_valid, foreign_request, prev_address_months_count,
current_address_months_count, days_since_request,
intended_balcon_amount, payment_type, housing_status,
date_of_birth_distinct_emails_4w, session_length_in_minutes, month

### Sentinel Values (-1 = missing, NOT a real number)
- prev_address_months_count = -1
- current_address_months_count = -1
- bank_months_count = -1

### Categorical Features (need one-hot encoding)
payment_type, employment_status, housing_status, device_os, source

---

## FEATURE ENGINEERING PLAN
1. has_prev_address: 1 if prev_address_months_count != -1 else 0
2. is_dirty_device: 1 if device_fraud_count > 0 else 0
3. velocity_ratio: velocity_6h / (velocity_4w + 1)
4. credit_to_income_ratio: proposed_credit_limit / (income + 1)
5. One-hot encode all categorical columns
6. MinMax scale: days_since_request, session_length_in_minutes,
   intended_balcon_amount

---

## MODELS
1. Random Forest — baseline, no tuning, raw imbalanced data
2. XGBoost — primary model
   - scale_pos_weight = 89.7
   - Compare with SMOTE oversampling
   - Hyperparameter tuning via grid search

---

## EVALUATION METRICS (NEVER use accuracy)
- Recall — primary metric
- Precision
- F1 Score
- Precision-Recall AUC
- ROC-AUC
- Gini Coefficient = 2 * AUC - 1
- KS Statistic (Kolmogorov-Smirnov)
- Confusion Matrix
- All evaluated at fixed 5% False Positive Rate

---

## MVP TARGET
XGBoost achieving at least 60% Recall at 5% FPR on test set

---

## PIPELINE PHASES
Phase 1 — Setup & Preprocessing (main/code/src/preprocess.py)
Phase 2 — EDA (main/code/notebooks/01_eda.ipynb)
Phase 3 — Modeling (main/code/src/train.py + main/code/notebooks/03_modeling.ipynb)
Phase 4 — Evaluation (main/code/src/evaluate.py)
Phase 5 — SHAP Explainability (main/code/src/explain.py)
Phase 6 — Streamlit App (main/code/app/streamlit_app.py)

---

## FINAL DELIVERABLE
Streamlit app:
- User inputs application features
- Model outputs fraud risk score (0-100%)
- Flag / Approve decision
- SHAP chart explaining why flagged
- Risk bands: 0-30% Low, 30-60% Medium, 60-100% High

---

## KEY DECISIONS MADE
- Temporal split only — no random splitting
- XGBoost over deep learning — tabular data, interpretable
- SHAP for explainability — required for MVP
- Gini + KS added to evaluation — industry standard metrics
- scale_pos_weight preferred over SMOTE — test both, pick better
- Base dataset only — no BAF variants

---

## REFERENCES
- Jesus et al. NeurIPS 2022 — https://arxiv.org/abs/2211.13358
- GitHub: https://github.com/feedzai/bank-account-fraud
- FTC Consumer Sentinel 2023
- Javelin Strategy & Research 2024
