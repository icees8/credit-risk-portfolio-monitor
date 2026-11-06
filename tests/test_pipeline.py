"""Comprehensive unit and integration tests for Credit Risk Portfolio Analytics."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.analytics import portfolio_kpis, segment_profile, cross_dimensional_cut, generate_portfolio_insights
from src.prepare_data import clean_customers, clean_loans, aggregate_bureau, aggregate_repayments
from src.ai_features import build_portfolio_evidence, fallback_grounded_answer, detect_anomalies, save_review_decision, load_review_decisions
from src.train_model import calculate_metrics, calculate_psi


# -------------------------------------------------------------
# 1. INGESTION & DATA PREPARATION TESTS
# -------------------------------------------------------------
def test_clean_customers_standardizes_gender():
    raw_df = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "gender": ["M", "Female", "F", "Male"],
        "age": [25, 40, 55, 65],
        "annual_income": [25000, 45000, 70000, 120000],
        "years_employed": [2.0, np.nan, 5.0, 10.0],
        "num_dependents": [np.nan, 1.0, 2.0, 0.0],
        "years_at_residence": [1.0, 3.0, np.nan, 8.0],
        "customer_since": ["2020-01-01", "2019-05-15", "2018-11-20", "2021-02-10"]
    })
    cleaned = clean_customers(raw_df)
    assert set(cleaned["gender"].unique()) <= {"Male", "Female"}
    assert cleaned["years_employed_missing_flag"].sum() == 1
    assert cleaned["years_employed"].isna().sum() == 0
    assert cleaned["income_band"].tolist() == ["<30k", "30-50k", "50-75k", "100k+"]


def test_clean_loans_imputes_interest_rate():
    raw_df = pd.DataFrame({
        "account_id": ["A1", "A2"],
        "customer_id": ["C1", "C2"],
        "interest_rate_pct": [12.5, np.nan],
        "origination_date": ["2023-01-15", "2023-06-20"]
    })
    cleaned = clean_loans(raw_df)
    assert cleaned["interest_rate_pct_missing_flag"].tolist() == [0, 1]
    assert cleaned["interest_rate_pct"].isna().sum() == 0
    assert cleaned["loan_vintage_year"].tolist() == [2023, 2023]


def test_aggregate_bureau_handles_active_and_written_off():
    raw_bureau = pd.DataFrame({
        "bureau_id": ["B1", "B2", "B3"],
        "customer_id": ["C1", "C1", "C2"],
        "account_status": ["Active", "Written-Off", "Closed"],
        "days_past_due": [0, 45, np.nan],
        "sanctioned_amount": [10000, 5000, 20000],
        "outstanding_amount": [6000, 4000, 0]
    })
    agg = aggregate_bureau(raw_bureau)
    c1 = agg[agg["customer_id"] == "C1"].iloc[0]
    assert c1["bureau_lines"] == 2
    assert c1["bureau_active_lines"] == 1
    assert c1["bureau_written_off_lines"] == 1
    assert c1["bureau_has_written_off"] == 1
    assert c1["bureau_max_dpd"] == 45


def test_aggregate_repayments_computes_delinquency_metrics():
    raw_rep = pd.DataFrame({
        "account_id": ["A1", "A1", "A1", "A1"],
        "payment_month": [1, 2, 10, 12],
        "emi_due": [500, 500, 500, 500],
        "emi_paid": [500, 250, 0, 500],
        "days_past_due": [0, 15, 65, 0],
        "payment_status": ["On-Time", "Late", "Missed", "On-Time"],
        "outstanding_balance": [10000, 9500, 5000, 4500]
    })
    agg = aggregate_repayments(raw_rep)
    a1 = agg[agg["account_id"] == "A1"].iloc[0]
    assert a1["repayment_months"] == 4
    assert a1["repayment_max_dpd"] == 65
    assert a1["repayment_missed_count"] == 2  # month 2 (paid < due) and month 10 (missed)
    assert a1["repayment_latest_outstanding"] == 4500


# -------------------------------------------------------------
# 2. PORTFOLIO ANALYTICS TESTS
# -------------------------------------------------------------
def test_portfolio_kpis():
    d = pd.DataFrame({
        "account_id": ["a", "b", "c", "d"],
        "target_default": [1, 0, 0, 1],
        "loan_amount": [10000, 20000, 30000, 40000],
        "repayment_latest_outstanding": [8000, 15000, 22000, 31000],
        "interest_rate_pct": [10.0, 12.0, 14.0, 16.0]
    })
    k = portfolio_kpis(d)
    assert k["accounts"] == 4
    assert k["defaults"] == 2
    assert k["default_rate_pct"] == 50.0
    assert k["original_loan_amount"] == 100000.0


def test_segment_profile():
    d = pd.DataFrame({
        "account_id": ["a", "b", "c"],
        "target_default": [1, 0, 0],
        "loan_amount": [100, 200, 300],
        "repayment_latest_outstanding": [80, 150, 200],
        "annual_income": [30000, 50000, 80000],
        "repayment_max_dpd": [30, 0, 0],
        "region": ["North", "North", "South"]
    })
    p = segment_profile(d, "region")
    north = p[p["region"] == "North"].iloc[0]
    assert north["accounts"] == 2
    assert north["defaults"] == 1
    assert north["default_rate_pct"] == 50.0


def test_generate_portfolio_insights_returns_5_insights():
    df_path = Path("data/processed/account_level_analytical_dataset.csv")
    if df_path.exists():
        df = pd.read_csv(df_path)
        insights = generate_portfolio_insights(df)
        assert len(insights) == 5
        for ins in insights:
            assert "finding" in ins and len(ins["finding"]) > 10
            assert "so_what" in ins and "Policy Action" in ins["so_what"]


# -------------------------------------------------------------
# 3. DEFAULT MODELING METRICS TESTS
# -------------------------------------------------------------
def test_discrimination_metrics():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    metrics = calculate_metrics(y_true, y_prob)
    assert metrics["auc"] == 1.0
    assert metrics["ks"] == 1.0
    assert metrics["gini"] == 1.0


def test_psi_calculation():
    baseline = np.random.normal(0.5, 0.1, 1000)
    shifted = np.random.normal(0.51, 0.1, 1000)
    psi = calculate_psi(baseline, shifted)
    assert psi >= 0.0
    assert psi < 0.10  # Mild shift should be in green range


# -------------------------------------------------------------
# 4. AI FEATURES TESTS
# -------------------------------------------------------------
def test_ask_the_portfolio_fallback_engine():
    evidence = {
        "portfolio_summary": {
            "active_filters": "All Accounts",
            "total_accounts": 1000,
            "total_defaults": 160,
            "portfolio_default_rate_pct": 16.0,
            "total_latest_outstanding_balance": 5000000.0,
            "total_origination_loan_amount": 6000000.0,
            "average_loan_amount": 6000.0
        },
        "by_loan_type": [
            {"segment": "Mortgage", "accounts": 200, "defaults": 50, "default_rate_pct": 25.0, "total_balance": 2000000.0},
            {"segment": "Auto Loan", "accounts": 400, "defaults": 40, "default_rate_pct": 10.0, "total_balance": 1500000.0}
        ],
        "by_region": [
            {"segment": "North", "accounts": 500, "defaults": 85, "default_rate_pct": 17.0, "total_balance": 2500000.0},
            {"segment": "South", "accounts": 500, "defaults": 75, "default_rate_pct": 15.0, "total_balance": 2500000.0}
        ],
        "by_employment_type": [
            {"segment": "Unemployed", "accounts": 100, "defaults": 35, "default_rate_pct": 35.0, "total_balance": 500000.0},
            {"segment": "Salaried", "accounts": 900, "defaults": 125, "default_rate_pct": 13.89, "total_balance": 4500000.0}
        ],
        "by_income_band": [
            {"segment": "<30k", "accounts": 100, "defaults": 38, "default_rate_pct": 38.0, "total_balance": 500000.0},
            {"segment": "100k+", "accounts": 200, "defaults": 18, "default_rate_pct": 9.0, "total_balance": 1500000.0}
        ]
    }
    # Test Loan Type query
    ans_loan = fallback_grounded_answer("Which loan type has the highest default rate?", evidence)
    assert "Mortgage" in ans_loan
    assert "25.00%" in ans_loan
    
    # Test Unanswerable Query
    ans_unsupported = fallback_grounded_answer("What is the weather in Tokyo?", evidence)
    assert "outside the scope" in ans_unsupported


def test_anomaly_detection_and_review_persistence(tmp_path):
    df_path = Path("data/processed/account_level_analytical_dataset.csv")
    if df_path.exists():
        df = pd.read_csv(df_path).head(100)
        flagged = detect_anomalies(df, contamination=0.05)
        assert len(flagged) > 0
        first_acc = flagged.iloc[0]["account_id"]
        assert len(flagged.iloc[0]["anomaly_reasons"]) > 0
        
        # Test decision saving
        save_review_decision(first_acc, "Confirmed Anomaly", notes="Test verification")
        decisions = load_review_decisions()
        assert first_acc in decisions
        assert decisions[first_acc]["decision"] == "Confirmed Anomaly"
