# Data Quality Decisions & Join Architecture

This document details the data ingestion, cleaning transformations, join grain logic, and feature engineering decisions implemented in `src/prepare_data.py` for the Credit Risk Portfolio Analytics solution.

---

## 1. Source Tables & Grain Mapping

The data environment consists of 5 synthetic banking tables representing ~10,000 customers and 14,685 loan accounts:

| Table | Source Grain | Rows | Key | Target Grain | Join Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `loan_accounts.csv` | 1 row per loan account | 14,685 | `account_id` (Primary Fact) | Account Grain | Fact table anchor (holds `target_default`) |
| `customers.csv` | 1 row per customer | 10,000 | `customer_id` | Customer Grain | Left merge `many_to_one` on `customer_id` |
| `bureau_credit.csv` | 1 row per external credit line | 22,979 | `customer_id` (Many-to-one) | Pre-aggregated to Customer | Group by `customer_id` -> merge `many_to_one` |
| `repayment_history.csv`| 1 row per account per month | 176,220 | `account_id` (12 months/acct) | Pre-aggregated to Account | Group by `account_id` -> merge `one_to_one` |
| `collections_events.csv`| 1 row per collection action | 8,635 | `account_id` (Delinquents only) | Pre-aggregated to Account | Group by `account_id` -> merge `one_to_one` |

---

## 2. Join Grain Preservation Logic

1. **Primary Fact Anchor (`loan_accounts.csv`)**:
   - Every loan account (`account_id`) represents a discrete credit decision holding the ground-truth prediction target `target_default` (portfolio default rate: 16.10%).
   - The final analytical dataset retains exactly 14,685 rows (100.0% retention) with no row duplication and zero dropped loans.

2. **Customer Join (`many_to_one` on `customer_id`)**:
   - A customer may hold multiple loans (e.g. Credit Card and Auto Loan).
   - Joining `customers` to `loan_accounts` on `customer_id` is strictly many-to-one.
   - Validation assertion confirms 100% of loans map to a valid customer profile.

3. **Bureau Credit Aggregation (`bureau_credit.csv`)**:
   - External credit lines exist at the customer level (22,979 lines across 8,992 customers).
   - **Crucial Rule**: Directly joining bureau lines to loans would result in severe row multiplication (Cartesian fan-out) and corrupt default rate calculations.
   - **Solution**: Bureau records are pre-aggregated to `customer_id` prior to joining, deriving customer-level credit history features.
   - Customers without external credit lines (10.0% thin-file borrowers) receive explicit zero-fill for credit counts/amounts and an indicator `bureau_record_flag = 0`.

4. **Repayment History Aggregation (`repayment_history.csv`)**:
   - Exactly 12 months of repayment history exist for each of the 14,685 accounts (176,220 total observations).
   - Pre-aggregated to `account_id` before joining to preserve the single-row-per-loan grain.

5. **Collections Events Aggregation (`collections_events.csv`)**:
   - Collections actions exist only for delinquent accounts (4,278 accounts; 29.1% coverage).
   - Aggregated by `account_id` to count recovery attempts, recovered capital, and escalated legal/settlement flags.
   - Accounts without collections activity receive zero counts and `has_collections_flag = 0`.

---

## 3. Data-Quality Issues & Remediation Decisions

### A. Inconsistent Gender Labels
- **Issue**: In `customers.csv`, ~6% of records were encoded as `"M"` or `"F"` instead of standard `"Male"` and `"Female"`.
- **Decision**: Standardized via explicit mapping `{"M": "Male", "F": "Female"}`. No customer rows dropped.

### B. Missing Values in Profile Fields
- **Issue**: `years_employed`, `num_dependents`, and `years_at_residence` each had 500 missing values (~5% missingness).
- **Decision**:
  1. Engineered boolean missingness indicator flags (`years_employed_missing_flag`, `num_dependents_missing_flag`, `years_at_residence_missing_flag`) to preserve credit-scoring signal associated with non-reporting.
  2. Imputed numerical values with median to preserve sample size and prevent model pipeline failures.

### C. Missing Interest Rates
- **Issue**: `loan_accounts.csv` had 294 missing values (2.0%) in `interest_rate_pct`.
- **Decision**:
  1. Added `interest_rate_pct_missing_flag`.
  2. Imputed missing values with the portfolio median interest rate.

### D. Bureau `date_closed` & `days_past_due` Nulls
- **Issue**: 11,243 bureau lines had null `date_closed`; 689 lines had null `days_past_due`.
- **Decision**:
  1. `date_closed` nulls are **not bad data**; they denote active, open external credit lines. Lines with `date_closed.isna()` are correctly aggregated into `bureau_active_lines`.
  2. Null `days_past_due` represents non-delinquent reporting and is treated as 0 DPD in aggregations.

---

## 4. Feature Engineering Architecture

### A. Repayment Behavior Features
- `repayment_months`: Observation tenure (12 months).
- `repayment_max_dpd`: Peak delinquency observed across 12 months.
- `repayment_mean_dpd`: Average monthly days past due.
- `repayment_missed_count` / `repayment_missed_rate`: Frequency and proportion of missed installments.
- `repayment_late_count` / `repayment_late_rate`: Frequency and proportion of delayed payments.
- `repayment_ontime_count`: Count of timely payments.
- `repayment_paid_ratio`: Cumulative EMI paid divided by cumulative EMI due (`sum(emi_paid) / sum(emi_due)`).
- `repayment_dpd_ge_30`, `repayment_dpd_ge_60`, `repayment_dpd_ge_90`: Milestone delinquency counts.
- `repayment_dpd_trend`: Trajectory metric subtracting early DPD (months 1-4) from recent DPD (months 9-12); positive values flag deteriorating distress.
- `repayment_latest_outstanding`: Month-12 closing balance.

### B. External Bureau Credit Features
- `bureau_lines`, `bureau_active_lines`, `bureau_closed_lines`: External credit line volume.
- `bureau_written_off_lines`: Prior charge-offs at other financial institutions.
- `bureau_total_sanctioned` & `bureau_total_outstanding`: Total external limit and balance.
- `bureau_utilization`: Aggregate bureau credit utilization (`outstanding / sanctioned`).
- `bureau_max_dpd` & `bureau_mean_dpd`: External historical delinquency severity.
- `bureau_has_written_off`: Binary flag indicating prior default at third-party lenders.
- `bureau_record_flag`: Distinguishes established credit history from thin-file applicants.

### C. Financial Distress & Solvency Ratios
- `dti_ratio`: Internal Debt-to-Income ratio (`loan_amount / annual_income`).
- `total_debt`: Cumulative exposure across internal loan and external bureau liabilities.
- `total_debt_to_income`: Total leverage across all institutions divided by annual income.
- `installment_to_income`: Monthly payment burden (`(loan_amount / tenure) / (annual_income / 12)`).
- `customer_relationship_years_at_origination`: Tenure of banking relationship prior to loan grant.

---

## 5. Output Verification Summary

```text
============================================================
DATA INGESTION & FEATURE ENGINEERING COMPLETE
Analytical dataset: data/processed/account_level_analytical_dataset.csv
Rows: 14,685 (100.0% retention)
Columns: 73
Target default rate: 16.10%
Customer match: 100.0%
Repayment match: 100.0%
Bureau match: 90.0% (10.0% thin-file cleanly zero-filled)
Collections match: 29.1% (70.9% non-delinquent cleanly zero-filled)
============================================================
```
