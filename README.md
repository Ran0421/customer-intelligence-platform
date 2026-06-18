# Customer Churn Prediction Platform

An end-to-end machine learning platform to predict telecom customer churn, explain individual risk factors, and surface actionable retention strategies.

---

## Overview

Customer churn is one of the costliest problems in the telecom industry. This project builds a full ML pipeline — from raw data to a deployed interactive application — that identifies at-risk customers, quantifies revenue exposure, and explains predictions at the individual level using SHAP.

---

## Project Structure

```
├── data/
│   ├── Telco-Customer-Churn.csv        # Raw dataset (7,043 records)
│   └── telco_cleaned.csv               # Cleaned dataset after preprocessing
├── notebooks/
│   ├── 01_data_understanding.ipynb     # Data cleaning, type fixes, imbalance analysis
│   ├── 02_eda.ipynb                    # EDA, churn drivers, revenue risk analysis
│   └── 03_modeling.ipynb              # Modeling, SMOTE, XGBoost, SHAP explainability
├── models/
│   ├── churn_model.pkl                 # Trained XGBoost model
│   ├── model_columns.pkl               # Feature column order for inference
│   └── shap_explainer.pkl              # SHAP TreeExplainer for real-time explanations
├── app/
│   └── app.py                          # Streamlit application
└── README.md
```

---

## Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Size:** 7,043 customer records, 21 features
- **Target:** `Churn` — binary (Yes / No), ~26.6% positive rate
- **Features:** Demographics, account info, service subscriptions, billing details

---

## Methodology

### 1. Data Understanding & Cleaning
- Fixed `TotalCharges` stored as object dtype; converted to numeric
- Removed 11 records with missing values
- Flagged class imbalance (~26.6% churn) — flagged accuracy as misleading metric

### 2. Exploratory Data Analysis
- Identified top churn drivers: low tenure, high monthly charges, month-to-month contracts, fiber optic service
- Revenue risk analysis: quantified monthly revenue at risk by contract type and tenure segment
- Customer segmentation: Early (0–12m), Mid (12–36m), Loyal (36m+)

### 3. Modeling
| Model | CV ROC-AUC | Churn Recall | Churn F1 |
|---|---|---|---|
| Logistic Regression | 0.81 | 0.76 | 0.59 |
| Random Forest | 0.82 | 0.69 | 0.60 |
| **XGBoost** | **0.84** | **0.76** | **0.59** |

- **Class imbalance:** SMOTE applied on training data only (no leakage); `class_weight='balanced'` for LR and RF; `scale_pos_weight` for XGBoost
- **Validation:** Stratified 5-fold cross-validation for all three models
- **Selected model:** XGBoost — highest CV ROC-AUC (0.84 ± 0.005), stable generalisation

### 4. Explainability (SHAP)
- Global feature importance via SHAP summary bar plot
- Feature impact direction via beeswarm plot
- Per-customer waterfall plot — explains exactly why a specific customer was flagged as high risk

---

## Key Findings

- **Tenure** is the strongest churn signal — customers in their first 12 months churn at the highest rate
- **Month-to-month contracts** show dramatically higher churn vs annual contracts
- **High monthly charges** correlate strongly with churn, especially for fiber optic subscribers
- **Revenue risk** is concentrated in early-tenure, month-to-month customers — the highest-priority retention segment

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Python, Pandas, NumPy |
| ML modeling | Scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Persistence | Joblib |

---

## Setup

```bash
git clone https://github.com/your-username/churn-prediction-platform.git
cd churn-prediction-platform
pip install -r requirements.txt
```

Run the Streamlit app:
```bash
streamlit run app/app.py
```

Run notebooks in order:
```
01_data_understanding.ipynb → 02_eda.ipynb → 03_modeling.ipynb
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
shap
matplotlib
seaborn
streamlit
joblib
```

---

## Results

- **XGBoost CV ROC-AUC:** 0.84 ± 0.005
- **Churn Recall:** 76% — correctly identifies 3 in 4 actual churners
- **Deployed:** Interactive Streamlit app with real-time prediction, risk segmentation, and SHAP-based individual explanations


Ranjeeta Mashal
