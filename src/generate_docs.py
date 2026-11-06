"""PDF Documentation Generator for Credit Risk Portfolio Monitor.

Generates:
1. docs/architecture.pdf: Exactly 1 page featuring the comprehensive architecture flow chart diagram:
   - The flow from raw data -> ingestion -> analytics -> model -> dashboard, with both AI features shown.
   - The tool or library used at each stage, where the LLM is called, and what data is sent to it.
2. docs/solution_overview.pdf: Exactly 1 page executive overview brief.
3. docs/comprehensive_project_guide.pdf: Master comprehensive guide explaining everything:
   approach, plain-English definitions of key terms (PD, AUC, KS, Gini, SHAP, PSI),
   data ingestion & join architecture, 5 empirical portfolio insights with 'So What's,
   supervised default modeling results, dual embedded AI features & governance,
   dashboard workflows, and bank enterprise production requirements.
"""
from pathlib import Path
import os
import pypdf
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image
)

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
FLOWCHART_PNG = DOCS_DIR / "architecture_flowchart.png"


# =====================================================================
# 1. GENERATE VISUAL ARCHITECTURE FLOW CHART DIAGRAM (LANDSCAPE)
# =====================================================================
def generate_architecture_flowchart_image(output_png: Path = FLOWCHART_PNG):
    """Render an institutional flow chart diagram in landscape to eliminate horizontal stretching."""
    fig, ax = plt.subplots(figsize=(11.0, 8.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # =================================================================
    # HEADER BANNER
    # =================================================================
    ax.text(50, 97.0, "CREDIT RISK PORTFOLIO MONITORING PLATFORM — SYSTEM ARCHITECTURE",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color="#0F172A")
    ax.text(50, 94.4, "End-to-End Pipeline: Raw Data → Ingestion → Analytics → Default Model → Dashboard → Dual Embedded AI Systems",
            ha="center", va="center", fontsize=7.2, fontweight="medium", color="#334155")
    ax.text(50, 92.4, "Tool/Library per Stage | Exact LLM Calling Endpoint & Integration Point | Strict Zero-PII Aggregate Payload Schema",
            ha="center", va="center", fontsize=6.2, fontstyle="italic", color="#2563EB")
    ax.plot([3.0, 97.0], [90.8, 90.8], color="#CBD5E1", linewidth=0.8)

    # =================================================================
    # TOP ROW: 5 PIPELINE STAGES (LEFT TO RIGHT)
    # =================================================================
    stage_w = 16.2
    gap = 3.25
    y_top = 47.5
    h_top = 41.5

    stages = [
        {
            "num": "STAGE 1",
            "title": "RAW DATA",
            "sub": "5 Relational CSVs",
            "tool": "Storage: data/raw/",
            "bg": "#F8FAFC", "border": "#64748B", "title_col": "#0F172A",
            "arrow_label": "Relational Tables",
            "items": [
                "• customers.csv (10k)",
                "  Key: customer_id",
                "  Demographics, income",
                "• loan_accounts.csv (14.7k)",
                "  Key: account_id (Anchor)",
                "  target_default (16.1%)",
                "• bureau_credit.csv (23k)",
                "  External balances, DPD",
                "• repayment_history.csv",
                "  176k 12m DPD records",
                "• collections_events.csv",
                "  8.6k recovery contacts"
            ]
        },
        {
            "num": "STAGE 2",
            "title": "DATA INGESTION",
            "sub": "Feature Engineering",
            "tool": "Python 3.13, Pandas",
            "bg": "#EFF6FF", "border": "#2563EB", "title_col": "#1E3A8A",
            "arrow_label": "Analytical Dataset",
            "items": [
                "• Clean & Standardize",
                "  Gender: Male/Female",
                "  Median imputation flags",
                "  Fixed charge-off regex",
                "• Pre-Aggregations",
                "  Bureau -> Customer",
                "  Repayments -> Account",
                "• 0 CARTESIAN FAN-OUT",
                "  14,685 rows preserved",
                "• 59 Predictive Signals",
                "  DTI, leverage, DPD trend",
                "• Output: analytical.csv"
            ]
        },
        {
            "num": "STAGE 3",
            "title": "ANALYTICS",
            "sub": "Profiling & Insights",
            "tool": "Pandas, NumPy, SciPy",
            "bg": "#FFFBEB", "border": "#D97706", "title_col": "#78350F",
            "arrow_label": "70/30 Splits",
            "items": [
                "• Portfolio KPIs",
                "  14,685 accounts",
                "  16.10% default rate",
                "  $512.6M original balance",
                "• 5-D Profiling Cuts",
                "  Income, Employment,",
                "  Region, Age, Loan Type",
                "• Bivariate Cross-Cut",
                "  Employment x Income",
                "  Unemployed <$30k: 44.1%",
                "• Policy Directives",
                "  35% DTI cap, LTV audits"
            ]
        },
        {
            "num": "STAGE 4",
            "title": "DEFAULT MODEL",
            "sub": "Supervised ML & SHAP",
            "tool": "Scikit-Learn, LightGBM",
            "bg": "#FAF5FF", "border": "#7C3AED", "title_col": "#4C1D95",
            "arrow_label": "PD & SHAP",
            "items": [
                "• Stratified 70/30 Holdout",
                "  10,279 train / 4,406 test",
                "• Baseline Logistic Reg",
                "  AUC: 0.6565 | KS: 0.2308",
                "• Random Forest (Champ)",
                "  AUC: 0.6609 | KS: 0.2342",
                "  Gini: 0.3217",
                "• Feature Importance & SHAP",
                "  Income, DPD, Leverage",
                "• Monitoring Plan",
                "  Baseline PSI: 0.0250",
                "  Drift trigger: PSI >= 0.20"
            ]
        },
        {
            "num": "STAGE 5",
            "title": "DASHBOARD",
            "sub": "Streamlit 1.67 App",
            "tool": "Streamlit, Plotly",
            "bg": "#F0FDFA", "border": "#0D9488", "title_col": "#134E4A",
            "arrow_label": "",
            "items": [
                "• Tab 1: Portfolio KPIs",
                "  Risk bands, vintages",
                "• Tab 2: Segmentation",
                "  Cross-cut heatmap",
                "• Tab 3: Account Dossier",
                "  Predicted PD & SHAP",
                "• Tab 4: Ask Portfolio",
                "  Conversational LLM",
                "• Tab 5: Anomaly Queue",
                "  Investigation dossier",
                "• Tab 6: Model Ops & PSI"
            ]
        }
    ]

    for idx, stg in enumerate(stages):
        sx = 3.0 + idx * (stage_w + gap)
        box = patches.FancyBboxPatch(
            (sx, y_top), stage_w, h_top, boxstyle="round,pad=0.3",
            facecolor=stg["bg"], edgecolor=stg["border"], linewidth=1.1, zorder=2
        )
        ax.add_patch(box)

        # Header in Box
        ax.text(sx + stage_w/2.0, y_top + h_top - 2.0, stg["num"],
                ha="center", va="center", fontsize=6.8, fontweight="bold", color=stg["border"], zorder=3)
        ax.text(sx + stage_w/2.0, y_top + h_top - 4.2, stg["title"],
                ha="center", va="center", fontsize=8.0, fontweight="bold", color=stg["title_col"], zorder=3)
        ax.text(sx + stage_w/2.0, y_top + h_top - 6.0, stg["sub"],
                ha="center", va="center", fontsize=5.8, fontstyle="italic", color="#475569", zorder=3)

        # Tool Badge
        tb = patches.FancyBboxPatch(
            (sx + 1.0, y_top + h_top - 8.8), stage_w - 2.0, 1.9,
            boxstyle="round,pad=0.15", facecolor="#FFFFFF", edgecolor=stg["border"], linewidth=0.6, zorder=3
        )
        ax.add_patch(tb)
        ax.text(sx + stage_w/2.0, y_top + h_top - 7.85, stg["tool"],
                ha="center", va="center", fontsize=5.0, fontweight="bold", color=stg["border"], zorder=4)

        ax.plot([sx + 0.8, sx + stage_w - 0.8], [y_top + h_top - 9.8, y_top + h_top - 9.8],
                color=stg["border"], linewidth=0.5, alpha=0.5, zorder=3)

        # Bullet items
        for l_idx, item in enumerate(stg["items"]):
            is_bold = "•" in item and ("CARTESIAN" in item or "Random Forest" in item or "Portfolio KPIs" in item or "Clean" in item or "Pre-Aggregations" in item)
            ax.text(sx + 1.2, y_top + h_top - 11.5 - l_idx * 2.4, item,
                    ha="left", va="center", fontsize=5.0,
                    fontweight="bold" if is_bold else "normal",
                    color=stg["title_col"] if is_bold else "#1E293B", zorder=3)

        # Rightward Connecting Arrow to next stage
        if idx < 4:
            ax.annotate("", xy=(sx + stage_w + gap - 0.2, y_top + h_top/2.0),
                        xytext=(sx + stage_w + 0.2, y_top + h_top/2.0),
                        arrowprops=dict(arrowstyle="->", lw=1.6, color="#2563EB"), zorder=4)
            lbl = stg["arrow_label"]
            if lbl:
                ax.text(sx + stage_w + gap/2.0, y_top + h_top/2.0 + 1.8, lbl,
                        ha="center", va="center", fontsize=4.4, fontweight="bold", color="#1D4ED8", zorder=5)

    # =================================================================
    # ORTHOGONAL BUS ROUTING TO DUAL AI FEATURES
    # =================================================================
    dash_cx = 3.0 + 4 * (stage_w + gap) + stage_w/2.0
    bus_y = 44.5
    ai1_cx = 26.0
    ai2_cx = 74.0

    ax.plot([dash_cx, dash_cx], [y_top, bus_y], color="#0D9488", linewidth=1.4, zorder=4)
    ax.plot([ai1_cx, dash_cx], [bus_y, bus_y], color="#0D9488", linewidth=1.4, zorder=4)

    # Downward and upward arrows
    ax.annotate("", xy=(ai1_cx, 41.5), xytext=(ai1_cx, bus_y),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"), zorder=4)
    ax.annotate("", xy=(ai1_cx + 2.5, bus_y), xytext=(ai1_cx + 2.5, 41.5),
                arrowprops=dict(arrowstyle="->", lw=1.1, ls="--", color="#059669"), zorder=4)

    ax.annotate("", xy=(ai2_cx, 41.5), xytext=(ai2_cx, bus_y),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#DC2626"), zorder=4)
    ax.annotate("", xy=(ai2_cx + 2.5, bus_y), xytext=(ai2_cx + 2.5, 41.5),
                arrowprops=dict(arrowstyle="->", lw=1.1, ls="--", color="#DC2626"), zorder=4)

    pill_b1 = patches.FancyBboxPatch((32.0, bus_y - 0.9), 23.0, 1.8, boxstyle="round,pad=0.2",
                                     facecolor="#ECFDF5", edgecolor="#6EE7B7", linewidth=0.6, zorder=5)
    ax.add_patch(pill_b1)
    ax.text(43.5, bus_y, "Query + Filtered Cohort Summary JSON", ha="center", va="center",
            fontsize=4.8, fontweight="bold", color="#065F46", zorder=6)

    pill_b2 = patches.FancyBboxPatch((57.0, bus_y - 0.9), 23.0, 1.8, boxstyle="round,pad=0.2",
                                     facecolor="#FEF2F2", edgecolor="#FCA5A5", linewidth=0.6, zorder=5)
    ax.add_patch(pill_b2)
    ax.text(68.5, bus_y, "Slice Multidimensional Feature Matrix", ha="center", va="center",
            fontsize=4.8, fontweight="bold", color="#991B1B", zorder=6)

    # =================================================================
    # BOTTOM ROW: DUAL AI FEATURES (SIDE-BY-SIDE PANELS)
    # =================================================================
    ai_w = 46.5
    ai_h = 39.5
    y_bot = 2.0

    # ---------------- AI FEATURE 1: ASK THE PORTFOLIO ----------------
    ai1_x = 3.0
    box_ai1 = patches.FancyBboxPatch(
        (ai1_x, y_bot), ai_w, ai_h, boxstyle="round,pad=0.4",
        facecolor="#F0FDF4", edgecolor="#16A34A", linewidth=1.2, zorder=2
    )
    ax.add_patch(box_ai1)

    ax.text(ai1_x + 1.8, y_bot + ai_h - 1.8, "AI FEATURE 1: 'ASK THE PORTFOLIO'",
            ha="left", va="center", fontsize=7.8, fontweight="bold", color="#14532D", zorder=3)
    ax.text(ai1_x + 1.8, y_bot + ai_h - 3.4, "Conversational LLM Assistant (Streamlit Tab 4)",
            ha="left", va="center", fontsize=5.6, fontstyle="italic", color="#15803D", zorder=3)
    
    badge1 = patches.FancyBboxPatch(
        (ai1_x + ai_w - 19.5, y_bot + ai_h - 3.4), 18.0, 1.9,
        boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#16A34A", linewidth=0.7, zorder=3
    )
    ax.add_patch(badge1)
    ax.text(ai1_x + ai_w - 10.5, y_bot + ai_h - 2.45, "Groq Cloud API | OpenAI SDK",
            ha="center", va="center", fontsize=5.0, fontweight="bold", color="#16A34A", zorder=4)

    ax.plot([ai1_x + 1.2, ai1_x + ai_w - 1.2], [y_bot + ai_h - 4.6, y_bot + ai_h - 4.6],
            color="#16A34A", linewidth=0.5, alpha=0.6, zorder=3)

    ax.text(ai1_x + 1.8, y_bot + ai_h - 6.0, "• OPERATIONAL TRIGGER: Risk analyst submits natural language prompt in Streamlit Tab 4.",
            ha="left", va="center", fontsize=5.2, fontweight="bold", color="#14532D", zorder=3)

    # Box: WHERE THE LLM IS CALLED
    box_where = patches.FancyBboxPatch(
        (ai1_x + 1.5, y_bot + 20.6), ai_w - 3.0, 11.6, boxstyle="round,pad=0.25",
        facecolor="#FFFFFF", edgecolor="#22C55E", linewidth=0.7, zorder=3
    )
    ax.add_patch(box_where)
    ax.text(ai1_x + 2.3, y_bot + 31.0, "WHERE THE LLM IS CALLED (Execution Point & Endpoint)",
            ha="left", va="center", fontsize=5.6, fontweight="bold", color="#15803D", zorder=4)
    where_lines = [
        "• Invoking Function: src/ai_features.py :: query_portfolio_llm()",
        "• Protocol: Outbound TLS 1.3 HTTPS POST request",
        "• Remote Endpoint: https://api.groq.com/openai/v1/chat/completions",
        "• Client SDK: openai.OpenAI / groq.Groq Python Client",
        "• Serving Model: openai/gpt-oss-20b (fallback: gpt-4o-mini)",
        "• Offline Fallback: Deterministic local analytical calculator executes",
        "  if GROQ_API_KEY / OPENAI_API_KEY is unset or network fails."
    ]
    for l_idx, line in enumerate(where_lines):
        ax.text(ai1_x + 2.3, y_bot + 29.5 - l_idx * 1.3, line,
                ha="left", va="center", fontsize=4.8, color="#1E293B", zorder=4)

    # Box: WHAT DATA IS SENT TO IT
    box_what = patches.FancyBboxPatch(
        (ai1_x + 1.5, y_bot + 8.4), ai_w - 3.0, 11.4, boxstyle="round,pad=0.25",
        facecolor="#FFFFFF", edgecolor="#22C55E", linewidth=0.7, zorder=3
    )
    ax.add_patch(box_what)
    ax.text(ai1_x + 2.3, y_bot + 18.6, "WHAT DATA IS SENT TO IT (Strict Aggregate Evidence Payload)",
            ha="left", va="center", fontsize=5.6, fontweight="bold", color="#15803D", zorder=4)
    what_lines = [
        "• Transmitted Data: ONLY Filtered Cohort Summary Evidence JSON:",
        "  - Active Filters: loan_type, region, employment, income_band",
        "  - Slice KPIs: total accounts (N), defaults, default_rate %, balance $",
        "  - Sub-segment Breakdown: default rates by loan type & income band",
        "  - Cross-cut Matrix: delinquency across employment x income",
        "• Payload Sample: {\"filters\":{\"loan_type\":\"All\"}, \"kpis\":{\"accounts\":14685,",
        "   \"defaults\":2364, \"default_rate\":\"16.10%\"}, \"by_product\":{...}}"
    ]
    for l_idx, line in enumerate(what_lines):
        is_code = "{\"filters\"" in line or "\"defaults\"" in line
        ax.text(ai1_x + 2.3, y_bot + 17.3 - l_idx * 1.25, line,
                ha="left", va="center", fontsize=4.3 if is_code else 4.7,
                fontfamily="monospace" if is_code else "sans-serif",
                color="#047857" if is_code else "#1E293B", zorder=4)

    # Privacy & Governance
    box_priv1 = patches.FancyBboxPatch(
        (ai1_x + 1.5, y_bot + 1.0), ai_w - 3.0, 6.6, boxstyle="round,pad=0.2",
        facecolor="#FEF2F2", edgecolor="#EF4444", linewidth=0.6, zorder=3
    )
    ax.add_patch(box_priv1)
    ax.text(ai1_x + 2.3, y_bot + 6.4, "ZERO-PII COMPLIANCE & HUMAN-IN-THE-LOOP (HITL)",
            ha="left", va="center", fontsize=5.2, fontweight="bold", color="#B91C1C", zorder=4)
    priv_lines1 = [
        "• ZERO Customer PII (No names, SSNs, phone numbers, or emails transmitted).",
        "• ZERO Account IDs or row-level records sent. Mathematical aggregates only.",
        "• Anti-hallucination prompt citations; unanswerable queries explicitly refused.",
        "• Human-in-the-Loop: Analyst verifies figures against KPI cards before memo use."
    ]
    for l_idx, line in enumerate(priv_lines1):
        ax.text(ai1_x + 2.3, y_bot + 5.1 - l_idx * 1.2, line,
                ha="left", va="center", fontsize=4.4, color="#991B1B", zorder=4)


    # ---------------- AI FEATURE 2: ANOMALY DETECTION ----------------
    ai2_x = 50.5
    box_ai2 = patches.FancyBboxPatch(
        (ai2_x, y_bot), ai_w, ai_h, boxstyle="round,pad=0.4",
        facecolor="#FFF7ED", edgecolor="#EA580C", linewidth=1.2, zorder=2
    )
    ax.add_patch(box_ai2)

    ax.text(ai2_x + 1.8, y_bot + ai_h - 1.8, "AI FEATURE 2: ANOMALY DETECTION",
            ha="left", va="center", fontsize=7.8, fontweight="bold", color="#7C2D12", zorder=3)
    ax.text(ai2_x + 1.8, y_bot + ai_h - 3.4, "Unsupervised Review Queue (Streamlit Tab 5)",
            ha="left", va="center", fontsize=5.6, fontstyle="italic", color="#C2410C", zorder=3)
    
    badge2 = patches.FancyBboxPatch(
        (ai2_x + ai_w - 19.5, y_bot + ai_h - 3.4), 18.0, 1.9,
        boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#EA580C", linewidth=0.7, zorder=3
    )
    ax.add_patch(badge2)
    ax.text(ai2_x + ai_w - 10.5, y_bot + ai_h - 2.45, "Scikit-Learn | IsolationForest",
            ha="center", va="center", fontsize=5.0, fontweight="bold", color="#EA580C", zorder=4)

    ax.plot([ai2_x + 1.2, ai2_x + ai_w - 1.2], [y_bot + ai_h - 4.6, y_bot + ai_h - 4.6],
            color="#EA580C", linewidth=0.5, alpha=0.6, zorder=3)

    ax.text(ai2_x + 1.8, y_bot + ai_h - 6.0, "• OPERATIONAL TRIGGER: Automated background execution on filtered cohort in Tab 5.",
            ha="left", va="center", fontsize=5.2, fontweight="bold", color="#7C2D12", zorder=3)

    # Box: DETECTION ENGINE
    box_det = patches.FancyBboxPatch(
        (ai2_x + 1.5, y_bot + 20.6), ai_w - 3.0, 11.6, boxstyle="round,pad=0.25",
        facecolor="#FFFFFF", edgecolor="#F97316", linewidth=0.7, zorder=3
    )
    ax.add_patch(box_det)
    ax.text(ai2_x + 2.3, y_bot + 31.0, "DETECTION ENGINE & MULTIDIMENSIONAL BENCHMARKING",
            ha="left", va="center", fontsize=5.6, fontweight="bold", color="#C2410C", zorder=4)
    det_lines = [
        "• Unsupervised Engine: Scikit-Learn IsolationForest (contamination=0.03)",
        "• Multidimensional Feature Matrix: annual_income, dti_ratio,",
        "  total_debt_to_income, repayment_max_dpd, repayment_mean_dpd, balance",
        "• Peer Group Normalization: Compares borrower against specific cohort",
        "  benchmarks (Loan Type x Income Band peer medians)",
        "• Trajectory Heuristic: Flags rapid DPD surges (late 6m vs early 6m)",
        "• Severity Ranking: Flagged accounts sorted by anomaly distance score."
    ]
    for l_idx, line in enumerate(det_lines):
        ax.text(ai2_x + 2.3, y_bot + 29.5 - l_idx * 1.3, line,
                ha="left", va="center", fontsize=4.8, color="#1E293B", zorder=4)

    # Box: REASON ATTRIBUTION
    box_attr = patches.FancyBboxPatch(
        (ai2_x + 1.5, y_bot + 8.4), ai_w - 3.0, 11.4, boxstyle="round,pad=0.25",
        facecolor="#FFFFFF", edgecolor="#F97316", linewidth=0.7, zorder=3
    )
    ax.add_patch(box_attr)
    ax.text(ai2_x + 2.3, y_bot + 18.6, "TRANSPARENT REASON ATTRIBUTION & INVESTIGATION DOSSIER",
            ha="left", va="center", fontsize=5.6, fontweight="bold", color="#C2410C", zorder=4)
    attr_lines = [
        "• Explainable Cause Attribution: System generates explicit trigger causes:",
        "  - Delinquency Outlier: 'Peak DPD 60d vs peer group median 8d (+52d)'",
        "  - Bureau Distress: 'External bureau write-offs present (Charge-off flag)'",
        "  - Excessive Leverage: 'DTI 8.4x vs peer group median 2.1x'",
        "  - Trajectory Surge: 'Late 6m DPD +45d higher than early 6m DPD'",
        "• Interactive Investigation Dossier: Side-by-side borrower profile vs peer",
        "  benchmarks across balance, terms, collections, and repayment curves."
    ]
    for l_idx, line in enumerate(attr_lines):
        ax.text(ai2_x + 2.3, y_bot + 17.3 - l_idx * 1.25, line,
                ha="left", va="center", fontsize=4.7, color="#1E293B", zorder=4)

    # Box: PERSISTENCE & GOVERNANCE
    box_audit = patches.FancyBboxPatch(
        (ai2_x + 1.5, y_bot + 1.0), ai_w - 3.0, 6.6, boxstyle="round,pad=0.2",
        facecolor="#F0FDF4", edgecolor="#16A34A", linewidth=0.6, zorder=3
    )
    ax.add_patch(box_audit)
    ax.text(ai2_x + 2.3, y_bot + 6.4, "DECISION AUDIT PERSISTENCE & HUMAN-IN-THE-LOOP MANDATE",
            ha="left", va="center", fontsize=5.2, fontweight="bold", color="#15803D", zorder=4)
    audit_lines = [
        "• Analyst Sign-Off: Selects Confirm / Dismiss / Escalate & enters notes.",
        "• Persistent Storage: data/processed/analyst_review_decisions.json (Audit Log).",
        "• Regulatory Compliance: Prevents automated credit line freezes without review.",
        "• Fair Lending: Provides compliant adverse action disclosure reasons (FCRA/ECOA)."
    ]
    for l_idx, line in enumerate(audit_lines):
        ax.text(ai2_x + 2.3, y_bot + 5.1 - l_idx * 1.2, line,
                ha="left", va="center", fontsize=4.4, color="#14532D", zorder=4)

    plt.tight_layout(pad=0.8)
    plt.savefig(output_png, dpi=300)
    plt.close()
    return output_png


# =====================================================================
# 2. BUILD docs/architecture.pdf (STRICTLY 1 PAGE, LANDSCAPE)
# =====================================================================
def build_architecture_pdf(output_path: Path):
    """Build the official 1-page architecture PDF featuring the full flowchart diagram in landscape."""
    generate_architecture_flowchart_image(FLOWCHART_PNG)

    im = PILImage.open(FLOWCHART_PNG)
    aspect = im.size[1] / im.size[0]

    w = 730
    h = 730 * aspect
    if h > 560:
        h = 560
        w = 560 / aspect

    target_path = output_path
    try:
        doc = SimpleDocTemplate(
            str(target_path),
            pagesize=landscape(letter),
            leftMargin=24,
            rightMargin=24,
            topMargin=20,
            bottomMargin=20
        )
        doc.build([Image(str(FLOWCHART_PNG), width=w, height=h)])
    except (PermissionError, OSError):
        target_path = output_path.with_name("architecture_landscape.pdf")
        print(f"Notice: {output_path.name} is currently open in your PDF viewer. Generated {target_path.name} (close the viewer to overwrite {output_path.name}).")
        doc = SimpleDocTemplate(
            str(target_path),
            pagesize=landscape(letter),
            leftMargin=24,
            rightMargin=24,
            topMargin=20,
            bottomMargin=20
        )
        doc.build([Image(str(FLOWCHART_PNG), width=w, height=h)])

    # Verification: Strictly 1 page
    reader = pypdf.PdfReader(str(target_path))
    assert len(reader.pages) == 1, f"{target_path.name} must be exactly 1 page, got {len(reader.pages)}"
    print(f"Generated {target_path} (EXACTLY {len(reader.pages)} page).")


# =====================================================================
# 3. BUILD docs/solution_overview.pdf (STRICTLY 1 PAGE)
# =====================================================================
def build_solution_overview_pdf(output_path: Path):
    """Build the official 1-page solution overview brief."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=24,
        rightMargin=24,
        topMargin=18,
        bottomMargin=18
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SolTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=16,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        "SolSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.0,
        leading=10,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=4
    )
    section_heading = ParagraphStyle(
        "SolSec",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=8.8,
        leading=11,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=3,
        spaceAfter=2
    )
    body_style = ParagraphStyle(
        "SolBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.9,
        leading=8.8,
        textColor=colors.HexColor("#2D3748")
    )
    cell_style = ParagraphStyle(
        "SolCell",
        parent=body_style,
        fontSize=6.4,
        leading=8.0
    )
    cell_head = ParagraphStyle(
        "SolCellHead",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1A365D")
    )

    story = []
    story.append(Paragraph("Credit Risk Portfolio Analytics | Solution Overview", title_style))
    story.append(Paragraph("Executive Brief: Dashboard Architecture, AI Governance, Default Modeling, Insights & Production Roadmap", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=3))

    # 1. Dashboard Views
    story.append(Paragraph("1. Dashboard Architecture & View-by-View Analyst Workflow", section_heading))
    dash_text = (
        "• <b>Portfolio Overview:</b> Core portfolio metrics (14,685 accounts, 16.10% default rate, $512.6M exposure), product delinquency bars, predicted PD risk bands (<10%, 10–25%, ≥25%), and monthly origination vintage curves.<br/>"
        "• <b>Segmentation & Insights:</b> Profile distributions across Income, Employment, Region, Age, Loan Type; plus interactive <b>Employment Type x Income Band</b> cross-tab heatmap.<br/>"
        "• <b>Account Risk & SHAP:</b> Searchable borrower dossier with predicted PD, terms, repayment history, and <b>local feature contributions (SHAP proxy)</b> explaining exact risk drivers.<br/>"
        "• <b>Ask the Portfolio (LLM):</b> Conversational interface answering slice queries without hallucinated figures.<br/>"
        "• <b>Anomaly Review Queue:</b> Case investigation queue with transparent trigger reasons and persistent decision recording.<br/>"
        "• <b>Model Monitoring:</b> Holdout discrimination curves, decile calibration (E/A), and post-deployment stability protocols."
    )
    story.append(Paragraph(dash_text, body_style))
    story.append(Spacer(1, 2))

    # 2. AI Governance Table
    story.append(Paragraph("2. Mandatory Embedded AI Features Governance Specification", section_heading))
    ai_table = [
        [
            Paragraph("<b>AI Feature</b>", cell_head),
            Paragraph("<b>Core Mechanism</b>", cell_head),
            Paragraph("<b>Operational Trigger</b>", cell_head),
            Paragraph("<b>False Positive (Risk)</b>", cell_head),
            Paragraph("<b>False Negative (Risk)</b>", cell_head),
            Paragraph("<b>Human-in-the-Loop Step</b>", cell_head)
        ],
        [
            Paragraph("<b>Ask the Portfolio</b> (LLM)", cell_head),
            Paragraph("Groq LLM (<code>openai/gpt-oss-20b</code>) queried over slice JSON aggregate metrics. Strict anti-hallucination prompt.", cell_style),
            Paragraph("Analyst types question and clicks 'Submit Query' in Tab 4.", cell_style),
            Paragraph("Model misinterprets slice denominator or asserts speculative causation.", cell_style),
            Paragraph("Model excessively declines a valid question inferable from slice data.", cell_style),
            Paragraph("Analyst verifies reported figures against displayed dashboard KPI cards before policy decisions.", cell_style)
        ],
        [
            Paragraph("<b>Anomaly Detection</b>", cell_head),
            Paragraph("Multivariate <code>IsolationForest</code> + peer group baselines (Loan Type x Income) + DPD trajectory heuristics.", cell_style),
            Paragraph("Automated background calculation on filtered cohort.", cell_style),
            Paragraph("Prime borrower with irregular seasonal cash flow flagged as credit distress.", cell_style),
            Paragraph("Gradual delinquency deteriorating just below contamination cut-off.", cell_style),
            Paragraph("Analyst inspects reason flags and logs <b>Confirm / Dismiss / Escalate</b> decision to audit log.", cell_style)
        ]
    ]
    t_ai = Table(ai_table, colWidths=[70, 115, 80, 100, 95, 104])
    t_ai.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EBF8FF")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FFFFFF")),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#F7FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_ai)
    story.append(Spacer(1, 2))

    # 3. Model Results & Monitoring
    story.append(Paragraph("3. Default Prediction Modeling Results & Post-Deployment Monitoring", section_heading))
    model_text = (
        "• <b>Methodology:</b> 70/30 stratified holdout (10,279 train / 4,406 test). Evaluated Logistic Regression baseline against Random Forest & LightGBM using 59 features spanning bureau history, 12-month repayment trajectory, collections activity, and borrower leverage.<br/>"
        "• <b>Discrimination Results:</b> <b>Random Forest achieved Test AUC: 0.6609, KS: 0.2342, Gini: 0.3217</b> (Baseline Logistic Regression achieved Test AUC: 0.6565, KS: 0.2308, Gini: 0.3131).<br/>"
        "• <b>Key Drivers:</b> Top features by Gini importance and SHAP are <code>annual_income</code> (0.064), <code>repayment_mean_dpd</code> (0.047), <code>years_employed</code> (0.045), <code>total_debt_to_income</code> (0.043), and <code>bureau_mean_dpd</code> (0.040).<br/>"
        "• <b>Monitoring Plan:</b> Monthly Population Stability Index (PSI) tracking (baseline-to-test PSI: 0.0250, well within <0.10 Green status). Automated alerts trigger if rolling 3-month AUC drops below 0.65, KS drops below 0.20, or score decile E/A calibration deviates beyond ±20%."
    )
    story.append(Paragraph(model_text, body_style))
    story.append(Spacer(1, 2))

    # 4. Top 3 Insights
    story.append(Paragraph("4. Top 3 Empirical Portfolio Insights & Strategic 'So What' Directives", section_heading))
    insights_text = (
        "<b>1. Steep Income Inversion Gradient (4.0x Risk Multiplier):</b> Borrowers earning <$30k default at <b>38.50%</b> (174/452) vs <b>9.53%</b> (293/3,076) for $100k+ earners. <i>So What:</i> Mandate max 35% DTI caps for sub-$30k applicants, require alternate income verification, and reduce unsecured credit limits.<br/>"
        "<b>2. Extreme Capital Exposure in Mortgages (24.65% Default Rate):</b> Mortgages demonstrate the highest product default rate (353/1,432) while carrying an average balance of $219,605 ($314.5M total original balance). <i>So What:</i> Conduct immediate collateral Loan-to-Value (LTV) audits, increase property appraisal haircuts, and tighten broker origination standards.<br/>"
        "<b>3. Bivariate Compounding Risk (Unemployed x Low Income Spike):</b> In the cross-dimensional intersection of Unemployed borrowers earning <$30k, default reaches <b>44.10%</b>, compared to <b>8.99%</b> for Salaried $100k+ borrowers. <i>So What:</i> Enforce bivariate underwriting stop-screens automatically rejecting unsecured credit for <$30k non-salaried applicants."
    )
    story.append(Paragraph(insights_text, body_style))
    story.append(Spacer(1, 2))

    # 5. Production Requirements
    story.append(Paragraph("5. Core Assumptions & Bank IT/Data Team Production Requirements", section_heading))
    prod_text = (
        "• <b>Assumptions:</b> Month-12 closing balance represents latest exposure; observation window is a standardized 12-month post-origination performance period; null bureau closed dates denote active open facilities; synthetic data distributions reflect underlying macroeconomic stability.<br/>"
        "• <b>IT & Data Requirements:</b> 1) <i>Point-in-Time Feature Store</i> (e.g. Feast/Hopsworks) for leak-free historical snapshots; 2) <i>Enterprise Secrets Vault</i> (e.g. AWS Secrets Manager / Azure Key Vault) for LLM API credentials; 3) <i>Durable Audit Database</i> (PostgreSQL) for regulatory storage of analyst review decisions; 4) <i>Model Risk Governance (SR 11-7)</i> approval and automated monthly PSI/CSI pipeline execution."
    )
    story.append(Paragraph(prod_text, body_style))

    doc.build(story)

    # Verification: Strictly 1 page
    reader = pypdf.PdfReader(str(output_path))
    assert len(reader.pages) == 1, f"solution_overview.pdf must be exactly 1 page, got {len(reader.pages)}"
    print(f"Generated {output_path} (EXACTLY {len(reader.pages)} page).")


# =====================================================================
# 4. BUILD docs/comprehensive_project_guide.pdf (MASTER MANUAL)
# =====================================================================
def build_comprehensive_project_guide(output_path: Path):
    """Generate the full, multi-page comprehensive project & approach guide."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    h1_style = ParagraphStyle(
        "GuideH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6,
        spaceBefore=10
    )
    h2_style = ParagraphStyle(
        "GuideH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=5
    )
    h3_style = ParagraphStyle(
        "GuideH3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#2C5282"),
        spaceBefore=8,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        "GuideBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.0,
        leading=12.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        "GuideBullet",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )
    cell_head = ParagraphStyle(
        "GuideCellHead",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.0,
        leading=10,
        textColor=colors.HexColor("#1A365D")
    )
    cell_text = ParagraphStyle(
        "GuideCellText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=9.8,
        textColor=colors.HexColor("#2D3748")
    )

    story = []

    # Cover Header
    story.append(Paragraph("Credit Risk Portfolio Analytics Platform", h1_style))
    story.append(Paragraph("Complete Technical, Methodological & Empirical Guide: Architecture, Key Concepts, Default Modeling, Insights & Dual AI Governance", ParagraphStyle("CoverSub", parent=body_style, fontSize=10.5, leading=14, textColor=colors.HexColor("#4A5568"))))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1A365D"), spaceAfter=12))

    # SECTION 1: EXECUTIVE OVERVIEW
    story.append(Paragraph("1. Executive Overview & Problem Formulation", h2_style))
    story.append(Paragraph(
        "Credit risk is the foundational risk assumed by lending institutions: the potential that a borrower will fail to meet their legal debt obligations. "
        "Modern banking demands systems that balance rigorous statistical discrimination with real-time operational monitoring, explainable machine learning, and secure artificial intelligence. "
        "This project establishes an institutional-grade credit monitoring platform built from 5 synthetic commercial bank tables (~10,000 customers, 14,685 loan accounts).",
        body_style
    ))
    story.append(Paragraph(
        "The solution delivers three core capabilities: "
        "(1) <b>High-integrity data ingestion & feature engineering</b> resolving relational fan-out traps and data quality defects; "
        "(2) <b>Supervised default prediction modeling</b> evaluating Logistic Regression and Random Forest architectures alongside Tree SHAP explainability and Population Stability Index (PSI) tracking; and "
        "(3) <b>Two embedded AI features</b> within an interactive Streamlit dashboard: a grounded natural-language assistant ('Ask the Portfolio') and a peer-comparative behavioral anomaly detection queue with human-in-the-loop audit logging.",
        body_style
    ))

    # SECTION 2: KEY TERMS & METRICS EXPLAINED IN SIMPLE ENGLISH
    story.append(Paragraph("2. Credit Risk & Machine Learning Demystified: Key Terms Explained Simply", h2_style))
    story.append(Paragraph("To ensure complete transparency across technical and executive stakeholders, below is a plain-English explanation of core concepts and formulas used in this project:", body_style))

    terms = [
        ("Default & Probability of Default (PD)",
         "A loan 'default' occurs when a borrower fails to pay installments as agreed for a sustained duration (conventionally 90+ days past due). "
         "The Probability of Default (PD) is a statistical number between 0% and 100% representing the model's estimate that an account will default within the observation window."),
        
        ("AUC (Area Under the ROC Curve)",
         "AUC measures the model's ranking ability. <i>Intuitive Analogy:</i> If you randomly pick one borrower who defaulted and one borrower who paid on time, "
         "AUC is the probability that your model assigned a higher risk score to the defaulted borrower. "
         "An AUC of 0.50 is equivalent to flipping a coin (zero signal); 1.0 represents psychic perfection. In retail credit risk, an AUC of 0.65–0.75 is typical, robust, and represents significant commercial value."),
        
        ("KS Statistic (Kolmogorov-Smirnov)",
         "The KS statistic measures the maximum percentage separation between the cumulative distribution of defaulters and non-defaulters across risk score deciles. "
         "<i>Intuitive Analogy:</i> Imagine lining up all defaulters from highest score to lowest, and doing the same for non-defaulters. "
         "KS is the widest gap between the two curves. A KS above 0.20 confirms the model cleanly separates good borrowers from distressed borrowers."),
        
        ("Gini Coefficient (2 * AUC - 1)",
         "Directly derived from the Lorenz curve, Gini is the global banking standard metric under the Basel Capital Accords. "
         "It rescales AUC to a 0 to 1 scale (Gini = 2 * AUC - 1). A model with AUC 0.6609 has a Gini of 0.3217, confirming solid rank-ordering discrimination."),
        
        ("SHAP (SHapley Additive exPlanations)",
         "Rooted in cooperative game theory (Nobel Prize in Economics), SHAP calculates the exact dollar or percentage contribution of each borrower attribute toward their final risk score. "
         "<i>Intuitive Analogy:</i> If an account has an 18% default risk while the portfolio average is 16%, SHAP tells you exactly which factors pushed it up (e.g. +4% from low income, +3% from missed payments) "
         "and which pushed it down (e.g. -5% from long employment tenure). This satisfies regulatory 'Adverse Action' notice requirements."),
        
        ("PSI (Population Stability Index)",
         "PSI measures whether the distribution of credit scores has drifted over time compared to the training baseline. "
         "Values < 0.10 indicate a stable population (Green); 0.10 to 0.25 indicate moderate shift (Yellow, trigger review); >= 0.25 indicate severe structural drift (Red, halt automated approvals and re-train)."),
        
        ("DTI (Debt-to-Income Ratio) & DPD (Days Past Due)",
         "DTI is the ratio of total loan liabilities to borrower annual income (higher DTI = higher debt burden). "
         "DPD tracks how many days have elapsed since a scheduled payment was due without being paid in full (e.g., 30 DPD, 60 DPD, 90 DPD).")
    ]

    for title, desc in terms:
        story.append(Paragraph(f"• <b>{title}:</b> {desc}", bullet_style))

    story.append(Spacer(1, 6))

    # SECTION 3: DATA INGESTION & QUALITY ARCHITECTURE
    story.append(Paragraph("3. Data Ingestion Architecture & Data Quality Decisions", h2_style))
    story.append(Paragraph(
        "Commercial banking data is inherently messy and distributed across relational tables with differing grains. "
        "A critical failure mode in credit engineering is the <b>'Cartesian Fan-Out Trap'</b>: naively merging many-to-one tables (like monthly repayments or external credit lines) directly onto loan accounts. "
        "This corrupts row counts, multiplies exposures, and invalidates statistical calculations.",
        body_style
    ))

    story.append(Paragraph("<b>Table Grain Architecture:</b>", h3_style))
    grain_table = [
        [Paragraph("<b>Table Name</b>", cell_head), Paragraph("<b>Source Grain</b>", cell_head), Paragraph("<b>Target Grain</b>", cell_head), Paragraph("<b>Aggregation & Join Strategy</b>", cell_head)],
        [Paragraph("loan_accounts.csv", cell_text), Paragraph("1 row / loan (14,685)", cell_text), Paragraph("Account Grain (Anchor)", cell_text), Paragraph("Anchor fact table holding target_default (16.10% default rate).", cell_text)],
        [Paragraph("customers.csv", cell_text), Paragraph("1 row / customer (10,000)", cell_text), Paragraph("Customer Grain", cell_text), Paragraph("Left merge many_to_one on customer_id (100% match rate).", cell_text)],
        [Paragraph("bureau_credit.csv", cell_text), Paragraph("1 row / line (22,979)", cell_text), Paragraph("Customer Grain", cell_text), Paragraph("Group by customer_id -> merge many_to_one. 10% thin-files cleanly zero-filled.", cell_text)],
        [Paragraph("repayment_history.csv", cell_text), Paragraph("1 row / acct / mo (176k)", cell_text), Paragraph("Account Grain", cell_text), Paragraph("Group by account_id -> merge one_to_one across 12 monthly cycles.", cell_text)],
        [Paragraph("collections_events.csv", cell_text), Paragraph("1 row / event (8,635)", cell_text), Paragraph("Account Grain", cell_text), Paragraph("Group by account_id -> merge one_to_one for 4,278 delinquent facilities.", cell_text)]
    ]
    t_grain = Table(grain_table, colWidths=[110, 100, 110, 220])
    t_grain.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EBF8FF")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#CBD5E0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_grain)
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Explicit Data Quality Decisions & Remediation:</b>", h3_style))
    story.append(Paragraph("1. <b>Inconsistent Gender Labels:</b> ~6% of records in <code>customers.csv</code> were encoded as 'M' or 'F' instead of 'Male' and 'Female'. Resolved via standard mapping <code>{'M': 'Male', 'F': 'Female'}</code>. No rows dropped.", bullet_style))
    story.append(Paragraph("2. <b>Missing Profile Values:</b> <code>years_employed</code>, <code>num_dependents</code>, and <code>years_at_residence</code> each had 500 missing values (~5%). Added explicit binary missingness indicator flags (e.g. <code>years_employed_missing_flag</code>) to capture non-reporting risk signals, then imputed medians.", bullet_style))
    story.append(Paragraph("3. <b>Missing Interest Rates:</b> 294 loan accounts had null interest rates. Flagged with <code>interest_rate_pct_missing_flag</code> and median-imputed.", bullet_style))
    story.append(Paragraph("4. <b>Bureau Active Line Logic:</b> 11,243 bureau lines had null <code>date_closed</code>. This was correctly recognized as <i>active open facilities</i> (not missing data) and aggregated into <code>bureau_active_lines</code>.", bullet_style))
    story.append(Paragraph("5. <b>Bureau Write-Off Bug Fix:</b> Resolved a subtle regex bug where <code>str.contains('write|settled')</code> failed to match <code>'Written-Off'</code>. Fixed with <code>str.contains('written|write|settled')</code>, restoring the critical prior write-off delinquency signal.", bullet_style))
    story.append(Paragraph("6. <b>Feature Catalog:</b> 59 engineered variables constructed across payment ratios (<code>repayment_paid_ratio</code>), delinquency trajectory (<code>repayment_dpd_trend</code>), bureau utilization (<code>bureau_utilization</code>), and leverage (<code>dti_ratio</code>, <code>total_debt_to_income</code>).", bullet_style))

    # SECTION 4: PORTFOLIO ANALYTICS & TOP 5 INSIGHTS
    story.append(Paragraph("4. Portfolio Analytics & Top 5 Empirical Insights ('So What' Actions)", h2_style))
    story.append(Paragraph(
        "A risk analytics dashboard must not merely display charts; it must translate empirical distributions into concrete underwriting policies. "
        "Analysis of the 14,685 loans reveals five profound structural patterns:",
        body_style
    ))

    insights = [
        ("1. Steep Income Inversion Gradient (4.0x Risk Multiplier)",
         "Annual Income",
         "Borrowers earning <$30k experience a 38.50% default rate (174 / 452 accounts), compared to just 9.53% for borrowers earning $100k+ (293 / 3,076 accounts).",
         "Policy Directive: Implement a hard Debt-to-Income cap of 35% for sub-$30k earners, mandate independent third-party income verification, and reduce uncollateralized lending limits for entry-level income cohorts."),
        
        ("2. Extreme Delinquency Among Unemployed Borrowers",
         "Employment Type",
         "Unemployed borrowers exhibit a 34.96% default rate (251 / 718 accounts), more than double the 14.97% observed in Salaried professionals (n=7,546).",
         "Policy Directive: Restrict unsecured personal loans and credit cards to applicants with active, verified employment. For existing unemployed accounts, trigger early forbearance and loan restructuring workflows before 30 DPD."),
        
        ("3. Severe Capital Concentration in High-Defaulting Mortgages",
         "Loan Product Type",
         "Mortgage facilities exhibit the highest product default rate at 24.65% (353 / 1,432 accounts), while carrying an average balance of $219,605 ($314.5M original balance). Auto Loans default at 13.88%.",
         "Policy Directive: Because mortgage loss severity is driven by high ticket size, conduct immediate Loan-to-Value (LTV) appraisal audits, tighten property collateral haircut guidelines, and review broker origination channels."),
        
        ("4. Young Borrower Vulnerability (18–25 Age Cohort)",
         "Age Cohort",
         "Borrowers aged 18–25 show a 22.16% default rate (427 / 1,927), over double the 10.56% observed in mature borrowers aged 56+ (n=833).",
         "Policy Directive: Younger borrowers lack credit seasoning and savings buffers. Implement 'starter' credit lines with automated step-up limit reviews contingent on 6 consecutive on-time monthly payments."),
        
        ("5. Compounding Cross-Cut: Unemployed x Low Income Spike (44.1% Default Rate)",
         "Employment x Income Cross-Cut",
         "In the cross-dimensional intersection of Unemployed borrowers earning <$30k, default rate reaches 44.10% (229 accounts). Conversely, Salaried borrowers earning $100k+ default at only 8.99% — an 8.9x risk divergence across quadrants.",
         "Policy Directive: Implement bivariate underwriting cut-offs. Automatically knock out loan applications combining <$30k income with Unemployed or Contract status unless secured by 100% liquid cash collateral.")
    ]

    for title, dim, finding, so_what in insights:
        story.append(Paragraph(f"<b>{title}</b> <i>[{dim}]</i>", h3_style))
        story.append(Paragraph(f"• <b>Empirical Finding:</b> {finding}", bullet_style))
        story.append(Paragraph(f"• <b>🎯 Strategic / Underwriting Impact ('So What'):</b> <code>{so_what}</code>", bullet_style))

    story.append(Spacer(1, 6))

    # SECTION 5: DEFAULT MODELING RESULTS & SHAP
    story.append(Paragraph("5. Supervised Default Prediction Modeling & Discrimination Results", h2_style))
    story.append(Paragraph(
        "To establish a reliable estimate of default probability (PD), we evaluated a regularized Logistic Regression baseline against Random Forest and LightGBM machine learning architectures. "
        "The model was trained on a <b>70/30 stratified holdout split</b> (10,279 train / 4,406 test accounts) using all 59 engineered features.",
        body_style
    ))

    model_table = [
        [Paragraph("<b>Model Architecture</b>", cell_head), Paragraph("<b>Train AUC</b>", cell_head), Paragraph("<b>Test AUC</b>", cell_head), Paragraph("<b>Test KS</b>", cell_head), Paragraph("<b>Test Gini</b>", cell_head), Paragraph("<b>Overfit Gap</b>", cell_head), Paragraph("<b>Deployment Status</b>", cell_head)],
        [Paragraph("<b>Random Forest Classifier</b>", cell_text), Paragraph("0.9150", cell_text), Paragraph("<b>0.6609</b>", cell_text), Paragraph("<b>0.2342</b>", cell_text), Paragraph("<b>0.3217</b>", cell_text), Paragraph("0.2541", cell_text), Paragraph("<b>Primary Production Model</b>", cell_text)],
        [Paragraph("<b>Logistic Regression Baseline</b>", cell_text), Paragraph("0.7009", cell_text), Paragraph("<b>0.6565</b>", cell_text), Paragraph("<b>0.2308</b>", cell_text), Paragraph("<b>0.3131</b>", cell_text), Paragraph("0.0444", cell_text), Paragraph("Regulatory Baseline", cell_text)],
        [Paragraph("<b>LightGBM Classifier</b>", cell_text), Paragraph("0.9074", cell_text), Paragraph("0.6453", cell_text), Paragraph("0.2114", cell_text), Paragraph("0.2906", cell_text), Paragraph("0.2621", cell_text), Paragraph("Challenger Benchmark", cell_text)]
    ]
    t_mod = Table(model_table, colWidths=[130, 60, 60, 60, 60, 70, 100])
    t_mod.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EBF8FF")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FFFFFF")),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#F7FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#CBD5E0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_mod)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Top 5 Predictive Drivers (Random Forest Gini Importance & Tree SHAP):</b><br/>"
        "1. <code>annual_income</code> (0.0644): Affordability anchor; lower income severely diminishes repayment cushion.<br/>"
        "2. <code>repayment_mean_dpd</code> (0.0473): Primary behavioral delinquency signal; past delinquency strongly predicts future charge-off.<br/>"
        "3. <code>years_employed</code> (0.0449): Proxy for job tenure and continuous cash flow stability.<br/>"
        "4. <code>total_debt_to_income</code> (0.0426): Aggregate multi-institutional indebtedness relative to capacity.<br/>"
        "5. <code>bureau_mean_dpd</code> (0.0398): External credit bureau track record across third-party lenders.",
        body_style
    ))

    # SECTION 6: DUAL EMBEDDED AI FEATURES & GOVERNANCE
    story.append(Paragraph("6. Dual Embedded AI Features & Enterprise Governance Blueprint", h2_style))
    story.append(Paragraph(
        "A core innovation of this platform is the integration of two distinct AI systems directly embedded into the decision workflows:",
        body_style
    ))

    story.append(Paragraph("<b>AI Feature 1: 'Ask the Portfolio' (Grounded LLM Assistant)</b>", h3_style))
    story.append(Paragraph(
        "• <b>Mechanism:</b> Risk analysts submit natural language questions in Tab 4. The application compiles an on-the-fly JSON evidence payload from the currently filtered cohort (containing slice account counts, defaults, segment default rates, and balances). "
        "This JSON is transmitted via HTTPS to Groq Cloud (<code>openai/gpt-oss-20b</code>) or OpenAI (<code>gpt-4o-mini</code>).<br/>"
        "• <b>Zero-PII Guarantee:</b> Under no circumstances are individual customer IDs, names, or row-level records sent to external APIs. Only aggregate counts and percentages are transmitted.<br/>"
        "• <b>Anti-Hallucination Guardrails:</b> The system prompt explicitly commands the model to use <i>only</i> the provided numbers, state denominators, cite source tables, and explicitly decline out-of-scope questions (macro forecasts, competitors).<br/>"
        "• <b>Deterministic Fallback:</b> If no API key is provided, the platform transparently falls back to an internal analytical query engine that calculates exact figures locally.",
        body_style
    ))

    story.append(Paragraph("<b>AI Feature 2: Behavioral Anomaly Detection & Review Queue</b>", h3_style))
    story.append(Paragraph(
        "• <b>Mechanism:</b> Combines Scikit-Learn <code>IsolationForest</code> (contamination=0.03) with peer-group comparative benchmarks (Loan Type x Income Tier medians) and trajectory heuristics (DPD surges).<br/>"
        "• <b>Transparent Reason Attribution:</b> Rather than an opaque black-box score, every flagged account is assigned human-readable triggers (e.g. 'Peak DPD 60d vs peer median 8d', '1 prior charged-off bureau line', 'DTI ratio 8.0x').<br/>"
        "• <b>Human-in-the-Loop Audit Trail:</b> Risk analysts review flagged accounts in Tab 5, investigate profile context, and submit formal <b>Confirm / Dismiss / Escalate</b> decisions with case notes. Decisions persist durably to <code>data/processed/analyst_review_decisions.json</code>.",
        body_style
    ))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Mandatory AI Governance Specification Matrix:</b>", h3_style))
    ai_gov_table = [
        [Paragraph("<b>AI Feature</b>", cell_head), Paragraph("<b>Trigger</b>", cell_head), Paragraph("<b>False Positive Risk</b>", cell_head), Paragraph("<b>False Negative Risk</b>", cell_head), Paragraph("<b>Human-in-the-Loop Step</b>", cell_head)],
        [
            Paragraph("<b>Ask the Portfolio (LLM)</b>", cell_text),
            Paragraph("Analyst enters question and clicks 'Submit Query'.", cell_text),
            Paragraph("LLM misinterprets slice denominator or asserts speculative causation.", cell_text),
            Paragraph("LLM excessively declines a valid question inferable from slice data.", cell_text),
            Paragraph("Analyst validates figures against dashboard KPI cards before policy memos.", cell_text)
        ],
        [
            Paragraph("<b>Anomaly Detection</b>", cell_text),
            Paragraph("Automated background calculation on filtered cohort.", cell_text),
            Paragraph("Prime borrower with irregular seasonal cash flow flagged as distressed.", cell_text),
            Paragraph("Gradual delinquency deteriorating just beneath contamination limit.", cell_text),
            Paragraph("Analyst investigates case dossier and logs persistent Confirm / Dismiss decision.", cell_text)
        ]
    ]
    t_gov = Table(ai_gov_table, colWidths=[90, 100, 120, 110, 120])
    t_gov.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EBF8FF")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#CBD5E0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_gov)
    story.append(Spacer(1, 6))

    # SECTION 7: DASHBOARD ARCHITECTURE & ANALYST WORKFLOWS
    story.append(Paragraph("7. Interactive Dashboard Design & User Workflows", h2_style))
    story.append(Paragraph(
        "The Streamlit dashboard (<code>app/app.py</code>) is organized into 6 specialized operational tabs:",
        body_style
    ))
    story.append(Paragraph("• <b>Tab 1: Portfolio Overview:</b> Executive summary cards (accounts, default rate %, exposure, origination volume), product default rate bars, risk band pie chart (<10%, 10–25%, ≥25%), and historical monthly vintage curves.", bullet_style))
    story.append(Paragraph("• <b>Tab 2: Segmentation & Insights:</b> Socio-demographic profiling across Income, Employment, Region, Age, Loan Type; plus the interactive <b>Employment Type x Income Band</b> cross-tab heatmap and the 5 empirical insight cards.", bullet_style))
    story.append(Paragraph("• <b>Tab 3: Account Risk & SHAP:</b> Searchable borrower dossier displaying predicted PD, loan terms, bureau history, 12-month repayment performance, and a <b>local risk driver waterfall chart</b> explaining individual risk contributions.", bullet_style))
    story.append(Paragraph("• <b>Tab 4: Ask the Portfolio (LLM):</b> Grounded conversational assistant with query preset pills, live API connectivity status, exact numerical citations, and out-of-scope refusal logic.", bullet_style))
    story.append(Paragraph("• <b>Tab 5: Anomaly Detection Queue:</b> Review queue of flagged behavioral outliers, highlighted reason badges, peer median comparisons, interactive decision recording, and a live decision audit table.", bullet_style))
    story.append(Paragraph("• <b>Tab 6: Model Monitoring & Governance:</b> Holdout discrimination metrics table, top 15 global feature importance chart, decile calibration (Expected vs Actual defaults), and PSI stability monitoring specs.", bullet_style))

    story.append(Spacer(1, 6))

    # SECTION 8: POST-DEPLOYMENT MONITORING & BANK IT ROADMAP
    story.append(Paragraph("8. Post-Deployment Monitoring & Enterprise Production Roadmap", h2_style))
    story.append(Paragraph(
        "In commercial banking, model deployment is merely Day 1. Regulatory compliance under <b>Federal Reserve SR 11-7 / OCC 2011-12 (Supervisory Guidance on Model Risk Management)</b> "
        "requires continuous monitoring for distribution drift and performance degradation.",
        body_style
    ))
    story.append(Paragraph("<b>1. Population Stability Index (PSI) Protocol:</b> Monthly automated score distribution comparison against training baseline. Current simulated test PSI is <code>0.0250</code> (Green: < 0.10). If PSI crosses 0.10, warning alerts trigger; if >= 0.25, automated approvals are paused.", bullet_style))
    story.append(Paragraph("<b>2. Discrimination Degradation Floors:</b> Automated monitoring tracks rolling 3-month holdout performance. If Test AUC drops below 0.65 or KS drops below 0.20, model re-estimation is automatically initiated.", bullet_style))
    story.append(Paragraph("<b>3. Decile Calibration Monitoring:</b> Expected-to-Actual (E/A) default ratios tracked by decile; deviations beyond ±20% trigger credit reserve adjustments.", bullet_style))
    story.append(Paragraph("<b>4. Bank IT & Data Engineering Production Requirements:</b>", h3_style))
    story.append(Paragraph("• <i>Point-in-Time Feature Store</i> (e.g. Feast / Hopsworks): To prevent future data leakage and guarantee point-in-time feature consistency across training and live scoring.", bullet_style))
    story.append(Paragraph("• <i>Enterprise Secrets Vault</i> (e.g. AWS Secrets Manager / Azure Key Vault / HashiCorp Vault): Secure storage and automated rotation of LLM API credentials without file-based .env exposure.", bullet_style))
    story.append(Paragraph("• <i>Durable Audit Database</i> (PostgreSQL / Snowflake): Scalable ACID-compliant persistence for analyst review decisions, timestamps, and justification notes for regulatory auditability.", bullet_style))
    story.append(Paragraph("• <i>Model Risk Governance (MRM) Sign-Off:</i> Independent validation review, developmental evidence documentation, and approval under supervisory SR 11-7 standards.", bullet_style))

    doc.build(story)
    reader = pypdf.PdfReader(str(output_path))
    print(f"Generated {output_path} ({len(reader.pages)} pages).")


def generate_all_documentation():
    build_architecture_pdf(DOCS_DIR / "architecture.pdf")
    build_solution_overview_pdf(DOCS_DIR / "solution_overview.pdf")
    build_comprehensive_project_guide(DOCS_DIR / "comprehensive_project_guide.pdf")


if __name__ == "__main__":
    generate_all_documentation()
