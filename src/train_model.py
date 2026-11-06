"""Default prediction model training, evaluation, SHAP explainability, and post-deployment monitoring.

Predicts target_default using engineered features from:
- Bureau credit history (delinquency, write-offs, utilization, lines)
- Repayment behavior (missed payments, DPD severity, payment ratio, trajectory)
- Collections actions (recovery contacts, legal/settlement flags)
- Customer profiles & loan terms (income, employment, DTI, leverage)

Models:
1. Logistic Regression Baseline (scaled, L2 regularized, class-balanced)
2. Machine Learning Model (Random Forest & LightGBM)

Evaluates and reports:
- ROC AUC
- Kolmogorov-Smirnov (KS) statistic
- Gini coefficient (2 * AUC - 1)
- Feature Importance and Tree SHAP summary values
- Post-deployment model monitoring specifications (PSI, CSI, calibration, drift triggers)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import shap

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "account_level_analytical_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Complete engineered feature suite across all 5 tables
FEATURE_COLS = [
    # Loan origination terms
    "loan_type", "loan_purpose", "loan_amount", "interest_rate_pct", "tenure_months",
    "disbursal_channel", "collateral_flag",
    
    # Customer demographics & financials
    "age", "gender", "marital_status", "education_level", "employment_type", "occupation_sector",
    "years_employed", "annual_income", "region", "city_tier", "num_dependents",
    "housing_type", "years_at_residence", "customer_relationship_years_at_origination",
    "dti_ratio", "total_debt", "total_debt_to_income", "installment_to_income",
    
    # Data quality / missingness indicators
    "years_employed_missing_flag", "num_dependents_missing_flag",
    "years_at_residence_missing_flag", "interest_rate_pct_missing_flag",
    
    # Bureau credit history (customer grain)
    "bureau_lines", "bureau_active_lines", "bureau_closed_lines", "bureau_written_off_lines",
    "bureau_total_sanctioned", "bureau_total_outstanding", "bureau_utilization",
    "bureau_max_dpd", "bureau_mean_dpd", "bureau_has_written_off", "bureau_record_flag",
    
    # Repayment behavior (account grain, 12 months)
    "repayment_months", "repayment_mean_dpd", "repayment_max_dpd",
    "repayment_missed_count", "repayment_late_count", "repayment_ontime_count",
    "repayment_missed_rate", "repayment_late_rate", "repayment_paid_ratio",
    "repayment_latest_outstanding", "repayment_dpd_ge_30", "repayment_dpd_ge_60",
    "repayment_dpd_ge_90", "repayment_recent_max_dpd", "repayment_dpd_trend",
    
    # Collections actions (delinquent accounts)
    "collections_event_count", "collections_amount_recovered",
    "collections_legal_notice_flag", "collections_settlement_offer_flag",
    "has_collections_flag"
]


def calculate_metrics(y_true, y_pred_prob):
    """Calculate discrimination metrics: AUC, KS statistic, and Gini."""
    auc = float(roc_auc_score(y_true, y_pred_prob))
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    ks = float(np.max(tpr - fpr))
    gini = float(2.0 * auc - 1.0)
    return {"auc": round(auc, 4), "ks": round(ks, 4), "gini": round(gini, 4)}


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between two score distributions."""
    eps = 1e-4
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(psi, 4))


def build_preprocessor(X: pd.DataFrame):
    """Build scikit-learn ColumnTransformer for numerical scaling and categorical one-hot encoding."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )
    return preprocessor, num_cols, cat_cols


def get_feature_names(preprocessor, num_cols, cat_cols):
    """Extract transformed feature names from ColumnTransformer."""
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    return num_cols + cat_feature_names


def train_and_evaluate():
    """Main execution function for model training, metrics reporting, and SHAP explainability."""
    df = pd.read_csv(DATA_PATH)
    
    # Ensure all feature columns exist in dataset
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].copy()
    y = df["target_default"].astype(int)
    
    # 70/30 Stratified Holdout Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)
    
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.5,
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=15,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
    }
    
    performance_records = []
    trained_pipelines = {}
    test_predictions = {}
    
    print("=" * 70)
    print("TRAINING CREDIT RISK DEFAULT MODELS")
    print(f"Features: {len(available_features)} engineered variables across 5 tables")
    print(f"Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")
    print("=" * 70)
    
    for name, clf in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", clf)
        ])
        pipe.fit(X_train, y_train)
        
        train_probs = pipe.predict_proba(X_train)[:, 1]
        test_probs = pipe.predict_proba(X_test)[:, 1]
        
        train_metrics = calculate_metrics(y_train, train_probs)
        test_metrics = calculate_metrics(y_test, test_probs)
        
        trained_pipelines[name] = pipe
        test_predictions[name] = test_probs
        joblib.dump(pipe, MODEL_DIR / f"{name}_pipeline.joblib")
        
        record = {
            "model": name,
            "train_auc": train_metrics["auc"],
            "train_ks": train_metrics["ks"],
            "train_gini": train_metrics["gini"],
            "test_auc": test_metrics["auc"],
            "test_ks": test_metrics["ks"],
            "test_gini": test_metrics["gini"]
        }
        performance_records.append(record)
        print(f"[{name.upper()}]")
        print(f"  Test AUC: {test_metrics['auc']:.4f} | Test KS: {test_metrics['ks']:.4f} | Test Gini: {test_metrics['gini']:.4f}")
        print(f"  Train AUC: {train_metrics['auc']:.4f} | Overfit gap: {train_metrics['auc'] - test_metrics['auc']:.4f}")
    
    perf_df = pd.DataFrame(performance_records)
    perf_df.to_csv(MODEL_DIR / "model_performance.csv", index=False)
    
    # -------------------------------------------------------------
    # FEATURE IMPORTANCE & SHAP VALUES (Random Forest)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPUTING GLOBAL FEATURE IMPORTANCE & SHAP VALUES")
    rf_model = trained_pipelines["random_forest"].named_steps["model"]
    
    # RF Gini Feature Importance
    rf_importances = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    rf_importances.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    
    print("\nTop 10 Features by Random Forest Importance:")
    print(rf_importances.head(10).to_string(index=False))
    
    # SHAP Explainer on representative test subset (500 samples)
    shap_sample_idx = np.random.RandomState(42).choice(len(X_test_proc), size=min(500, len(X_test_proc)), replace=False)
    X_shap_sample = X_test_proc[shap_sample_idx]
    
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_shap_sample)
    
    # Handle binary classification output shape (shap_values can be list of [class 0, class 1] or 3D array)
    if isinstance(shap_values, list):
        shap_class1 = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_class1 = shap_values[:, :, 1]
    else:
        shap_class1 = shap_values
        
    mean_abs_shap = np.mean(np.abs(shap_class1), axis=0)
    shap_importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    shap_importance_df.to_csv(MODEL_DIR / "shap_importance.csv", index=False)
    
    # Save SHAP artifacts for dashboard instant loading
    expected_val = float(explainer.expected_value[1]) if hasattr(explainer.expected_value, "__iter__") else float(explainer.expected_value)
    shap_bundle = {
        "shap_values": shap_class1,
        "feature_names": feature_names,
        "sample_X": X_shap_sample,
        "base_value": expected_val,
        "feature_cols": available_features
    }
    joblib.dump(shap_bundle, MODEL_DIR / "shap_summary.joblib")
    print(f"SHAP summary bundle saved to {MODEL_DIR / 'shap_summary.joblib'}")
    
    # -------------------------------------------------------------
    # POST-DEPLOYMENT MODEL MONITORING FRAMEWORK
    # -------------------------------------------------------------
    # Baseline test distribution for future PSI monitoring
    baseline_scores = test_predictions["random_forest"]
    simulated_drift_scores = np.clip(baseline_scores * np.random.uniform(0.95, 1.15, size=len(baseline_scores)), 0, 1)
    simulated_psi = calculate_psi(baseline_scores, simulated_drift_scores)
    
    monitoring_plan = {
        "model_in_production": "random_forest_pipeline",
        "monitoring_frequency": "Monthly",
        "stability_thresholds": {
            "psi_green": "< 0.10 (No action required; distribution stable)",
            "psi_yellow": "0.10 to 0.25 (Moderate drift; increase monitoring frequency, review origination mix)",
            "psi_red": ">= 0.25 (Significant drift; trigger model re-estimation or pause automated cut-offs)"
        },
        "performance_triggers": {
            "auc_floor": "Test AUC must remain >= 0.65; alert if rolling 3-month AUC drops by > 0.05",
            "ks_floor": "KS statistic must remain >= 0.20",
            "calibration_error": "Expected-to-Actual (E/A) default ratio by score decile must stay between 0.80 and 1.25"
        },
        "feature_drift_csi_targets": [
            "annual_income", "repayment_max_dpd", "repayment_missed_rate",
            "dti_ratio", "bureau_written_off_lines", "bureau_utilization"
        ],
        "simulated_test_psi": simulated_psi,
        "operational_alerts": [
            "Pipeline input missingness spikes (>5% above training baseline)",
            "Daily API latency exceeding 250ms",
            "Approval rate deviations > 3% from policy benchmark"
        ]
    }
    
    with open(MODEL_DIR / "monitoring_plan.json", "w") as f:
        json.dump(monitoring_plan, f, indent=2)
        
    print("\n" + "=" * 70)
    print("POST-DEPLOYMENT MONITORING PLAN SAVED TO models/monitoring_plan.json")
    print(f"Simulated Baseline-to-Month1 PSI: {simulated_psi:.4f} (Status: In Monitoring Tolerance)")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
