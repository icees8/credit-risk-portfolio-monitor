# Credit Risk Portfolio Analytics & Monitoring Dashboard

An institutional credit-risk analytics and monitoring platform built from 5 synthetic bank tables (~10,000 customers, 14,685 loan accounts). Features portfolio profiling, multi-dimensional segmentation, supervised default prediction modeling (AUC, KS, Gini, SHAP), interactive exploratory dashboarding, and two embedded AI features: a grounded LLM "Ask the Portfolio" assistant and peer-comparative anomaly detection with human-in-the-loop decision auditing.

---

## 1. Python Version
**Python 3.13** (compatible with Python 3.11+).

---

## 2. Run Instructions (Three Commands)
Run the entire platform from installation to data preparation, model training, and dashboard launch in **three commands**:

```bash
pip install -r requirements.txt
python -m src.prepare_data && python -m src.train_model
streamlit run app/app.py
```

*(Optional: To run the automated test suite, execute `python -m pytest tests/`)*.

---

## 3. LLM API Key Configuration & Behavior When Missing

### How to Set the Key
- **Local Development**: Copy `.env.example` to `.env` and set your key:
  ```bash
  cp .env.example .env
  # Set GROQ_API_KEY=gsk_... or OPENAI_API_KEY=sk-...
  ```
- **Hosted Deployment (Streamlit Community Cloud / Hugging Face Spaces)**: Add `GROQ_API_KEY = "gsk_..."` or `OPENAI_API_KEY = "sk-..."` in the platform's **Secrets Management** settings.

### What the App Does if No Key is Set
If no API key is detected in the environment or secrets store:
1. **Zero Disruption**: The application runs completely without crashing, errors, or empty states.
2. **Transparent Notification**: The "Ask the Portfolio" assistant displays an informational badge:
   `"No API key detected in .env or secrets; operating using the deterministic grounded fallback engine."`
3. **Deterministic Grounded Fallback**: The assistant routes analyst questions directly to the underlying aggregate calculation engine in `src/ai_features.py`. It computes exact numbers (counts, observed default rates, dollar balances) from the active filtered portfolio slice, cites the source table, and explicitly declines out-of-scope questions (macro forecasts, competitors, unanswerable queries).

---

## 4. Repository Layout & Brief Mapping

| Assignment Requirement | Primary File / Directory | Description |
| :--- | :--- | :--- |
| **1. Data Ingestion & Quality** | [`src/prepare_data.py`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/src/prepare_data.py)<br/>[`docs/data_quality_and_joins.md`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/docs/data_quality_and_joins.md) | Ingestion pipeline merging all 5 tables at the loan-account grain, handling gender encoding, null imputation, bureau/repayment aggregation, and preserving 100% of accounts. |
| **2. Portfolio Analytics** | [`src/analytics.py`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/src/analytics.py) | Portfolio KPIs, discrete segment profiling across 5 dimensions, bivariate cross-cuts (Employment × Income), and 5 empirical insights with "So What" actions. |
| **3. Default Model & SHAP** | [`src/train_model.py`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/src/train_model.py)<br/>[`models/`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/models/) | Supervised default prediction using 59 engineered features from bureau and repayment history. Evaluates Logistic Regression baseline vs Random Forest & LightGBM. Reports AUC, KS, Gini, Tree SHAP, and post-deployment PSI monitoring. |
| **4. Interactive Dashboard** | [`app/app.py`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/app/app.py) | Full Streamlit application featuring Portfolio Overview, Segmentation, Account Risk & SHAP dossier, Ask the Portfolio, Anomaly Queue, and Model Governance tabs. |
| **5. AI Feature: Ask Portfolio** | [`src/ai_features.py`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/src/ai_features.py)<br/>[`app/app.py`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/app/app.py) (Tab 4) | Grounded Q&A over active filtered portfolio JSON evidence. Cites exact figures, handles unanswerable queries, and connects to Groq (`openai/gpt-oss-20b`) or deterministic fallback. |
| **6. AI Feature: Anomaly Detection**| [`src/ai_features.py`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/src/ai_features.py)<br/>[`app/app.py`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/app/app.py) (Tab 5) | Behavioral outlier detection via Isolation Forest and peer-group benchmarking. Transparent reason attribution and persistent human-in-the-loop review decision logging. |
| **7. Architecture Diagram (1p)** | [`docs/architecture.pdf`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/docs/architecture.pdf) | Exactly 1-page PDF featuring the visual system architecture diagram illustrating the end-to-end data flow (raw data -> ingestion -> modeling -> dashboard -> dual AI features), tools/libraries, where LLM is called, and payload. |
| **8. Solution Overview (1p)** | [`docs/solution_overview.pdf`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/docs/solution_overview.pdf) | Exactly 1-page PDF summarizing dashboard views, AI governance specifications, model results, top 3 insights, assumptions, and bank production requirements. |
| **9. Comprehensive Master Guide** | [`docs/comprehensive_project_guide.pdf`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/docs/comprehensive_project_guide.pdf) | Detailed 4-page master guide explaining the entire approach in plain English, defining key terms (PD, AUC, KS, Gini, SHAP, PSI), data decisions, insights with "So What"s, and bank IT roadmap. |
| **10. Document Generator** | [`src/generate_docs.py`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/src/generate_docs.py) | Reproducible ReportLab & Matplotlib generator ensuring PDFs strictly conform to brief and 1-page constraints. |
| **11. Testing Suite** | [`tests/`](file:///d:/job_hunt/Essex%20Test/credit_risk_submission_repo/tests/) | 13 automated unit and integration tests verifying joins, analytics, modeling metrics, PSI calculations, and AI features. |

---

## 5. Modeling Results Summary

Evaluated on a **70/30 Stratified Holdout Split** (10,279 training / 4,406 test accounts):

| Model Architecture | Train AUC | Test AUC | Test KS Statistic | Test Gini (2*AUC - 1) | Overfit Gap | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest Classifier** | 0.9150 | **0.6609** | **0.2342** | **0.3217** | 0.2541 | **Primary Production Model** |
| **Logistic Regression Baseline** | 0.7009 | **0.6565** | **0.2308** | **0.3131** | 0.0444 | Regulatory Baseline |
| **LightGBM Classifier** | 0.9074 | 0.6453 | 0.2114 | 0.2906 | 0.2621 | Benchmark Challenger |

### Top 5 Predictive Drivers (Random Forest & SHAP)
1. `annual_income` (Importance: 0.0644): Strongest affordability anchor.
2. `repayment_mean_dpd` (Importance: 0.0473): Primary behavioral delinquency signal.
3. `years_employed` (Importance: 0.0449): Job stability and cash flow continuity.
4. `total_debt_to_income` (Importance: 0.0426): Aggregate borrower leverage across institutions.
5. `bureau_mean_dpd` (Importance: 0.0398): Third-party credit track record.

---

## 6. Top Empirical Portfolio Insights & "So What" Directives

1. **Steep Income Inversion Gradient (4.0x Risk Multiplier)**:
   - *Finding*: Borrowers earning <$30k default at **38.50%** (174/452), whereas borrowers earning $100k+ default at **9.53%** (293/3,076).
   - *So What*: Underwriting must cap Debt-to-Income (DTI) at 35% for sub-$30k applicants and require verified bank statement cash flows rather than stated income.
2. **Capital Concentration in High-Defaulting Mortgages**:
   - *Finding*: Mortgage loans carry the highest default rate among all products at **24.65%** (353/1,432), with an average exposure of $219,605 ($314.5M total original balance). Auto Loans default at only 13.88%.
   - *So What*: Severe dollar loss severity risk. Credit risk committees must audit collateral Loan-to-Value (LTV) ratios, apply higher appraisal haircuts, and tighten broker origination criteria.
3. **Compounding Delinquency Spike (Unemployed × Low Income = 44.1% Default)**:
   - *Finding*: In the bivariate cross-cut of Unemployed borrowers earning <$30k, default reaches **44.10%**, compared to **8.99%** for Salaried $100k+ earners (an 8.9x risk spread).
   - *So What*: Enforce automated stop-screens in loan origination systems to reject unsecured credit applications combining <$30k income with non-salaried employment.

---

## 7. Dual Embedded AI Features Governance

```
                                      AI GOVERNANCE MATRIX
┌──────────────────────┬───────────────────────────────┬───────────────────────────────┬───────────────────────────────┐
│ AI Feature           │ Operational Trigger           │ Error Modes (FP / FN)         │ Human-in-the-Loop Step        │
├──────────────────────┼───────────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Ask the Portfolio    │ Analyst submits natural       │ FP: LLM misinterprets slice   │ Analyst validates numbers and │
│ (Grounded LLM Q&A)   │ language query in Tab 4.      │     denominator or causality. │ denominators against dashboard│
│                      │                               │ FN: Excessive decline of an   │ metric cards before memo use. │
│                      │                               │     inferable portfolio slice.│                               │
├──────────────────────┼───────────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ Anomaly Detection &  │ Automated Isolation Forest    │ FP: Prime borrower with sea-  │ Analyst investigates flags,   │
│ Review Queue         │ and peer trajectory rules     │     sonal income cash flow.   │ inspects borrower dossier, and│
│                      │ on slice behavioral metrics.  │ FN: Subtle distress staying   │ logs persistent Confirm /     │
│                      │                               │     beneath outlier cutoff.   │ Dismiss / Escalate decision.  │
└──────────────────────┴───────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

---

## 8. Post-Deployment Model Monitoring Framework
- **Population Stability Index (PSI)**: Monthly score distribution stability tracking against training baseline (simulated baseline-to-test PSI: `0.0246`, well below the 0.10 alert threshold).
- **Discrimination Floor Alerts**: Automated alert triggers if rolling 3-month AUC degrades below 0.65 or KS drops below 0.20.
- **Score Decile Calibration**: Expected-to-Actual (E/A) default ratio tracking by decile with ±20% tolerance bounds.
- **Characteristic Stability (CSI)**: Monthly feature drift monitoring across `annual_income`, `repayment_max_dpd`, `dti_ratio`, and `bureau_written_off_lines`.
