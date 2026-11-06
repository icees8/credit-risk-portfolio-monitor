"""Data ingestion and feature engineering pipeline for Credit Risk Portfolio Monitor.

Loads all five synthetic bank tables, resolves known data-quality issues,
constructs aggregated bureau and repayment behavior features, and produces
one clean, analysis-ready analytical dataset at the loan-account grain.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_FILE = PROCESSED_DIR / "account_level_analytical_dataset.csv"


def load_raw_tables(raw_dir: Path = RAW_DIR):
    """Load the five raw CSV tables."""
    customers = pd.read_csv(raw_dir / "customers.csv")
    loans = pd.read_csv(raw_dir / "loan_accounts.csv")
    bureau = pd.read_csv(raw_dir / "bureau_credit.csv")
    repayments = pd.read_csv(raw_dir / "repayment_history.csv")
    collections = pd.read_csv(raw_dir / "collections_events.csv")
    return customers, loans, bureau, repayments, collections


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    """Clean customers table:
    1. Standardize inconsistent gender encoding ('M' -> 'Male', 'F' -> 'Female').
    2. Add missingness indicators for years_employed, num_dependents, years_at_residence.
    3. Impute missing values (median for numeric).
    4. Create standardized demographic segments: income_band, age_band.
    5. Convert customer_since to datetime.
    """
    df = customers.copy()
    
    # 1. Standardize gender
    df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"}).astype(str)
    
    # 2. Track missingness before imputation
    for col in ["years_employed", "num_dependents", "years_at_residence"]:
        df[f"{col}_missing_flag"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median())
        
    # 3. Demographic bands
    df["income_band"] = pd.cut(
        df["annual_income"],
        bins=[0, 30000, 50000, 75000, 100000, np.inf],
        labels=["<30k", "30-50k", "50-75k", "75-100k", "100k+"]
    ).astype(str)
    
    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 200],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"]
    ).astype(str)
    
    # 4. Dates
    df["customer_since"] = pd.to_datetime(df["customer_since"], errors="coerce")
    return df


def clean_loans(loans: pd.DataFrame) -> pd.DataFrame:
    """Clean loan_accounts table:
    1. Track and impute missing interest_rate_pct.
    2. Convert origination_date to datetime and extract vintage year/month.
    """
    df = loans.copy()
    df["interest_rate_pct_missing_flag"] = df["interest_rate_pct"].isna().astype(int)
    df["interest_rate_pct"] = df["interest_rate_pct"].fillna(df["interest_rate_pct"].median())
    
    df["origination_date"] = pd.to_datetime(df["origination_date"], errors="coerce")
    df["loan_vintage_year"] = df["origination_date"].dt.year
    df["loan_vintage_month"] = df["origination_date"].dt.month
    return df


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Aggregate external bureau credit lines to the customer grain.
    Note: date_closed nulls denote active open lines (intentional domain logic).
    Missing days_past_due filled with 0 (no delinquency recorded).
    """
    b = bureau.copy()
    b["days_past_due"] = pd.to_numeric(b["days_past_due"], errors="coerce").fillna(0)
    b["sanctioned_amount"] = pd.to_numeric(b["sanctioned_amount"], errors="coerce").fillna(0)
    b["outstanding_amount"] = pd.to_numeric(b["outstanding_amount"], errors="coerce").fillna(0)
    
    agg = b.groupby("customer_id").agg(
        bureau_lines=("bureau_id", "count"),
        bureau_active_lines=("account_status", lambda s: (s.str.lower() == "active").sum()),
        bureau_closed_lines=("account_status", lambda s: (s.str.lower() == "closed").sum()),
        bureau_written_off_lines=("account_status", lambda s: s.str.lower().str.contains("written|write|settled", regex=True).sum()),
        bureau_total_sanctioned=("sanctioned_amount", "sum"),
        bureau_total_outstanding=("outstanding_amount", "sum"),
        bureau_max_dpd=("days_past_due", "max"),
        bureau_mean_dpd=("days_past_due", "mean")
    ).reset_index()
    
    agg["bureau_utilization"] = (
        agg["bureau_total_outstanding"] / agg["bureau_total_sanctioned"].replace(0, np.nan)
    ).fillna(0.0).clip(upper=5.0)
    agg["bureau_has_written_off"] = (agg["bureau_written_off_lines"] > 0).astype(int)
    agg["bureau_record_flag"] = 1
    return agg


def aggregate_repayments(repayments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly repayment behavior to the account grain.
    Constructs missed payment frequency, delinquency severity (DPD 30/60/90),
    payment ratio, latest outstanding balance, and delinquency trajectory.
    """
    r = repayments.copy()
    r["emi_due"] = pd.to_numeric(r["emi_due"], errors="coerce").fillna(0)
    r["emi_paid"] = pd.to_numeric(r["emi_paid"], errors="coerce").fillna(0)
    r["days_past_due"] = pd.to_numeric(r["days_past_due"], errors="coerce").fillna(0)
    r["outstanding_balance"] = pd.to_numeric(r["outstanding_balance"], errors="coerce").fillna(0)
    
    r["is_missed"] = ((r["payment_status"].str.lower() == "missed") | (r["emi_paid"] < r["emi_due"])).astype(int)
    r["is_late"] = (r["payment_status"].str.lower() == "late").astype(int)
    r["is_ontime"] = (r["payment_status"].str.lower() == "on-time").astype(int)
    r["dpd_ge_30"] = (r["days_past_due"] >= 30).astype(int)
    r["dpd_ge_60"] = (r["days_past_due"] >= 60).astype(int)
    r["dpd_ge_90"] = (r["days_past_due"] >= 90).astype(int)
    
    # Trajectory: early (months 1-4) vs late (months 9-12)
    early = r[r["payment_month"] <= 4].groupby("account_id").agg(early_max_dpd=("days_past_due", "max"))
    recent = r[r["payment_month"] >= 9].groupby("account_id").agg(
        recent_max_dpd=("days_past_due", "max"),
        recent_missed_count=("is_missed", "sum")
    )
    
    agg = r.sort_values("payment_month").groupby("account_id").agg(
        repayment_months=("payment_month", "count"),
        repayment_mean_dpd=("days_past_due", "mean"),
        repayment_max_dpd=("days_past_due", "max"),
        repayment_missed_count=("is_missed", "sum"),
        repayment_late_count=("is_late", "sum"),
        repayment_ontime_count=("is_ontime", "sum"),
        repayment_dpd_ge_30=("dpd_ge_30", "sum"),
        repayment_dpd_ge_60=("dpd_ge_60", "sum"),
        repayment_dpd_ge_90=("dpd_ge_90", "sum"),
        repayment_total_emi_due=("emi_due", "sum"),
        repayment_total_emi_paid=("emi_paid", "sum"),
        repayment_latest_outstanding=("outstanding_balance", "last")
    ).reset_index()
    
    agg = agg.merge(early, on="account_id", how="left").merge(recent, on="account_id", how="left")
    agg["early_max_dpd"] = agg["early_max_dpd"].fillna(0)
    agg["recent_max_dpd"] = agg["recent_max_dpd"].fillna(0)
    agg["recent_missed_count"] = agg["recent_missed_count"].fillna(0)
    
    agg["repayment_missed_rate"] = agg["repayment_missed_count"] / agg["repayment_months"].replace(0, np.nan)
    agg["repayment_late_rate"] = agg["repayment_late_count"] / agg["repayment_months"].replace(0, np.nan)
    agg["repayment_paid_ratio"] = (
        agg["repayment_total_emi_paid"] / agg["repayment_total_emi_due"].replace(0, np.nan)
    ).fillna(0.0).clip(upper=2.0)
    agg["repayment_dpd_trend"] = agg["recent_max_dpd"] - agg["early_max_dpd"]
    return agg


def aggregate_collections(collections: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sparse collections activity to the account grain."""
    c = collections.copy()
    c["amount_recovered"] = pd.to_numeric(c["amount_recovered"], errors="coerce").fillna(0)
    
    agg = c.groupby("account_id").agg(
        collections_event_count=("collection_event_id", "count"),
        collections_amount_recovered=("amount_recovered", "sum"),
        collections_legal_notice_flag=("action_type", lambda s: (s == "Legal Notice").sum()),
        collections_settlement_offer_flag=("action_type", lambda s: (s == "Settlement Offer").sum())
    ).reset_index()
    agg["has_collections_flag"] = 1
    return agg


def build_analytical_dataset(raw_dir: Path = RAW_DIR, out_file: Path = OUT_FILE) -> pd.DataFrame:
    """Executes the full join and feature engineering pipeline:
    1. Loads the 5 raw tables.
    2. Cleans customers and loan accounts.
    3. Aggregates bureau (customer grain), repayments (account grain), collections (account grain).
    4. Performs validated joins to preserve the exact account-level grain (1 row per loan).
    5. Engineers financial ratios and missingness flags.
    6. Persists the clean analytical dataset to disk.
    """
    customers_raw, loans_raw, bureau_raw, repayments_raw, collections_raw = load_raw_tables(raw_dir)
    
    customers = clean_customers(customers_raw)
    loans = clean_loans(loans_raw)
    bureau_agg = aggregate_bureau(bureau_raw)
    repayments_agg = aggregate_repayments(repayments_raw)
    collections_agg = aggregate_collections(collections_raw)
    
    # Fact join: loan_accounts is primary (14,685 rows)
    # Join customers (many-to-one on customer_id)
    df = loans.merge(customers, on="customer_id", how="left", validate="many_to_one")
    
    # Join bureau aggregates (many loans can belong to one customer)
    df = df.merge(bureau_agg, on="customer_id", how="left", validate="many_to_one")
    # For customers without bureau records, cleanly fill defaults:
    bureau_cols_zero = [
        "bureau_lines", "bureau_active_lines", "bureau_closed_lines",
        "bureau_written_off_lines", "bureau_total_sanctioned", "bureau_total_outstanding",
        "bureau_max_dpd", "bureau_mean_dpd", "bureau_utilization",
        "bureau_has_written_off", "bureau_record_flag"
    ]
    df[bureau_cols_zero] = df[bureau_cols_zero].fillna(0)
    
    # Join repayment aggregates (one-to-one on account_id)
    df = df.merge(repayments_agg, on="account_id", how="left", validate="one_to_one")
    
    # Join collections aggregates (one-to-one on account_id)
    df = df.merge(collections_agg, on="account_id", how="left", validate="one_to_one")
    col_cols_zero = [
        "collections_event_count", "collections_amount_recovered",
        "collections_legal_notice_flag", "collections_settlement_offer_flag",
        "has_collections_flag"
    ]
    df[col_cols_zero] = df[col_cols_zero].fillna(0)
    
    # Derived composite features and financial ratios
    df["customer_relationship_years_at_origination"] = (
        (df["origination_date"] - df["customer_since"]).dt.days / 365.25
    ).clip(lower=0.0).fillna(0.0)
    
    df["dti_ratio"] = (df["loan_amount"] / df["annual_income"].replace(0, np.nan)).fillna(0.0).clip(upper=10.0)
    df["total_debt"] = df["loan_amount"] + df["bureau_total_outstanding"]
    df["total_debt_to_income"] = (df["total_debt"] / df["annual_income"].replace(0, np.nan)).fillna(0.0).clip(upper=20.0)
    
    estimated_monthly_installment = df["loan_amount"] / df["tenure_months"].replace(0, np.nan)
    monthly_income = df["annual_income"] / 12.0
    df["installment_to_income"] = (estimated_monthly_installment / monthly_income.replace(0, np.nan)).fillna(0.0).clip(upper=5.0)
    
    # Quality validations
    assert len(df) == len(loans), f"Row count mismatch: expected {len(loans)}, got {len(df)}"
    assert df["account_id"].nunique() == len(df), "Duplicate account IDs detected"
    assert df["target_default"].isna().sum() == 0, "Missing values detected in target_default"
    
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    
    print("=" * 60)
    print("DATA INGESTION & FEATURE ENGINEERING COMPLETE")
    print(f"Analytical dataset saved to: {out_file}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print(f"Account retention: 100.0% ({df['account_id'].nunique():,}/{len(loans):,})")
    print(f"Target default rate: {df['target_default'].mean():.2%}")
    print(f"Customer join match rate: {df['age'].notna().mean():.1%}")
    print(f"Repayment history match rate: {df['repayment_months'].notna().mean():.1%}")
    print(f"Bureau history match rate: {df['bureau_record_flag'].mean():.1%}")
    print(f"Collections record rate: {df['has_collections_flag'].mean():.1%}")
    print("=" * 60)
    return df


if __name__ == "__main__":
    build_analytical_dataset()
