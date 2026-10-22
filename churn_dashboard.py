"""
Customer Churn Dashboard — Streamlit App
Run: streamlit run churn_dashboard.py

Required packages:
    pip install streamlit pandas numpy plotly scikit-learn
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── COLOUR PALETTE ────────────────────────────────────────────────────────────
GREEN_DARK  = "#0d6e3f"
GREEN_MID   = "#22a86a"
GREEN_LIGHT = "#56d9a0"
BLUE_DARK   = "#0a3d6b"
BLUE_MID    = "#1565c0"
BLUE_LIGHT  = "#42a5f5"
BG_PAGE     = "#f0f7f4"
BG_CARD     = "#ffffff"
TEXT_MAIN   = "#0d2618"
TEXT_MUTED  = "#4a7060"
DANGER      = "#c0392b"

PALETTE     = [GREEN_MID, BLUE_MID, GREEN_LIGHT, BLUE_LIGHT,
               GREEN_DARK, BLUE_DARK, "#80cbc4", "#a5d6a7"]

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    /* ---- page background ---- */
    .stApp {{ background-color: {BG_PAGE}; }}

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {BLUE_DARK} 0%, {GREEN_DARK} 100%);
    }}
    section[data-testid="stSidebar"] * {{ color: #e8f5e9 !important; }}
    section[data-testid="stSidebar"] .stFileUploader label {{ color: #b2dfdb !important; }}

    /* ---- header banner ---- */
    .dash-header {{
        background: linear-gradient(135deg, {BLUE_DARK} 0%, {GREEN_DARK} 100%);
        padding: 28px 36px 20px;
        border-radius: 12px;
        margin-bottom: 24px;
        color: #fff;
    }}
    .dash-header h1 {{ font-size: 2rem; font-weight: 700; margin: 0; color: #fff; }}
    .dash-header p  {{ font-size: 0.95rem; margin: 6px 0 0; color: #b2dfdb; }}

    /* ---- metric cards ---- */
    .metric-row {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
    .metric-card {{
        background: {BG_CARD};
        border-left: 5px solid {GREEN_MID};
        border-radius: 10px;
        padding: 18px 22px;
        flex: 1 1 160px;
        box-shadow: 0 1px 6px rgba(0,0,0,.06);
    }}
    .metric-card.blue  {{ border-left-color: {BLUE_MID}; }}
    .metric-card.alert {{ border-left-color: {DANGER}; }}
    .metric-card .label {{ font-size: 0.75rem; color: {TEXT_MUTED}; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }}
    .metric-card .value {{ font-size: 1.9rem; font-weight: 700; color: {TEXT_MAIN}; line-height: 1.1; }}
    .metric-card .sub   {{ font-size: 0.8rem; color: {TEXT_MUTED}; margin-top: 2px; }}

    /* ---- section headings ---- */
    .section-title {{
        font-size: 1rem; font-weight: 700; color: {BLUE_DARK};
        border-bottom: 2px solid {GREEN_LIGHT};
        padding-bottom: 6px; margin-bottom: 16px;
    }}

    /* ---- insight callout ---- */
    .insight {{
        background: #e8f5e9; border-left: 4px solid {GREEN_MID};
        border-radius: 6px; padding: 12px 16px;
        font-size: 0.88rem; color: {GREEN_DARK};
        margin-top: 10px;
    }}
    .insight.blue {{
        background: #e3f2fd; border-left-color: {BLUE_MID};
        color: {BLUE_DARK};
    }}

    /* ---- plotly chart containers ---- */
    .chart-card {{
        background: {BG_CARD};
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 6px rgba(0,0,0,.06);
        margin-bottom: 20px;
    }}

    /* ---- data table ---- */
    .stDataFrame {{ border-radius: 8px; overflow: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def fig_layout(fig, title=""):
    """Apply consistent styling to every Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=BLUE_DARK)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT_MAIN),
        margin=dict(t=40, b=20, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e0ece6", linecolor="#c8ddd4")
    fig.update_yaxes(showgrid=True, gridcolor="#e0ece6", linecolor="#c8ddd4")
    return fig


def metric_card(label, value, sub="", variant="green"):
    cls = "metric-card" if variant == "green" else f"metric-card {variant}"
    return f"""
    <div class="{cls}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>"""


def detect_churn_column(df: pd.DataFrame) -> str | None:
    """Return the most likely churn column name."""
    candidates = ["churn", "churned", "attrition", "exited", "left", "cancelled",
                  "churn_flag", "is_churn", "churn_label", "target"]
    for col in df.columns:
        if col.strip().lower() in candidates:
            return col
    return None


def binarise(series: pd.Series) -> pd.Series:
    """Convert Yes/No, True/False, 1/0 etc. to 0/1 ints."""
    s = series.astype(str).str.strip().str.lower()
    return s.map({"1": 1, "yes": 1, "true": 1, "y": 1,
                  "0": 0, "no":  0, "false": 0, "n": 0}).fillna(0).astype(int)


def load_sample() -> pd.DataFrame:
    """Generate a realistic synthetic churn dataset."""
    rng = np.random.default_rng(42)
    n = 1_000
    tenure    = rng.integers(1, 72, n)
    age       = rng.integers(18, 70, n)
    monthly   = rng.uniform(20, 120, n).round(2)
    total     = (monthly * tenure * rng.uniform(0.85, 1.05, n)).round(2)
    num_prods = rng.integers(1, 5, n)
    contracts = rng.choice(["Month-to-month", "One year", "Two year"], n,
                           p=[0.55, 0.25, 0.20])
    support   = rng.choice(["Yes", "No"], n, p=[0.45, 0.55])
    internet  = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.44, 0.34, 0.22])
    gender    = rng.choice(["Male", "Female"], n)
    region    = rng.choice(["North", "South", "East", "West"], n)

    # Churn probability biased realistically
    logit = (
        -3.5
        - 0.05 * tenure
        + 0.02 * monthly
        + 0.5  * (contracts == "Month-to-month")
        - 0.3  * (support == "Yes")
        + 0.4  * (internet == "Fiber optic")
    )
    prob  = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "CustomerID":       [f"C{i:05d}" for i in range(n)],
        "Gender":           gender,
        "Age":              age,
        "Region":           region,
        "Tenure":           tenure,
        "Contract":         contracts,
        "InternetService":  internet,
        "TechSupport":      support,
        "NumProducts":      num_prods,
        "MonthlyCharges":   monthly,
        "TotalCharges":     total,
        "Churn":            churn,
    })


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📁 Data Source")
    uploaded = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Must contain a churn/target column (0/1, Yes/No, True/False).",
    )
    st.markdown("---")
    st.markdown("### ⚙️ Options")
    run_model = st.checkbox("Run churn prediction model", value=True)
    test_size = st.slider("Model test split %", 10, 40, 20) / 100
    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.78rem;opacity:.7'>Green/Blue palette · "
        "Powered by Streamlit & Plotly</span>",
        unsafe_allow_html=True,
    )

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        data_source = f"**{uploaded.name}** ({len(df_raw):,} rows)"
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
else:
    df_raw = load_sample()
    data_source = "Sample dataset (1,000 synthetic customers)"

df = df_raw.copy()

# Detect churn column
churn_col = detect_churn_column(df)
if churn_col is None:
    with st.sidebar:
        churn_col = st.selectbox(
            "Select the churn / target column",
            options=df.columns.tolist(),
        )

df["_churn"] = binarise(df[churn_col])

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="dash-header">
        <h1>📉 Customer Churn Dashboard</h1>
        <p>Data: {data_source} &nbsp;|&nbsp; Churn column: <strong>{churn_col}</strong></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI METRICS ───────────────────────────────────────────────────────────────
total      = len(df)
churned    = df["_churn"].sum()
retained   = total - churned
churn_rate = churned / total * 100

# Numeric columns for averages
num_cols = df.select_dtypes(include="number").columns.difference(["_churn"]).tolist()

avg_tenure_label, avg_tenure_val = "", ""
if any("tenure" in c.lower() for c in num_cols):
    tc = next(c for c in num_cols if "tenure" in c.lower())
    avg_churned  = df[df["_churn"] == 1][tc].mean()
    avg_retained = df[df["_churn"] == 0][tc].mean()
    avg_tenure_label = f"Avg tenure (churned vs retained)"
    avg_tenure_val   = f"{avg_churned:.0f} vs {avg_retained:.0f} mo"

monthly_col = next((c for c in num_cols if "monthly" in c.lower() or "charge" in c.lower()), None)
avg_charge  = df[monthly_col].mean() if monthly_col else None

cards_html = "<div class='metric-row'>"
cards_html += metric_card("Total Customers", f"{total:,}", "in dataset")
cards_html += metric_card("Churned", f"{churned:,}", f"{churn_rate:.1f}% of total", "alert")
cards_html += metric_card("Retained", f"{retained:,}", f"{100-churn_rate:.1f}% of total", "blue")
if avg_charge:
    cards_html += metric_card(
        f"Avg {monthly_col}", f"${avg_charge:.2f}", "per customer / month"
    )
if avg_tenure_val:
    cards_html += metric_card(avg_tenure_label, avg_tenure_val, "months")
cards_html += "</div>"
st.markdown(cards_html, unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_overview, tab_segments, tab_model, tab_data = st.tabs(
    ["Overview", "Segment Analysis", "Prediction Model", "Raw Data"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    col1, col2 = st.columns(2)

    # Churn donut
    with col1:
        st.markdown("<div class='section-title'>Churn split</div>", unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=["Retained", "Churned"],
            values=[retained, churned],
            hole=0.55,
            marker_colors=[GREEN_MID, DANGER],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,}<extra></extra>",
        ))
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            annotations=[dict(
                text=f"<b>{churn_rate:.1f}%</b><br>churn",
                x=0.5, y=0.5, font_size=16,
                font_color=BLUE_DARK, showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Churn by categorical columns
    with col2:
        cat_cols = df.select_dtypes(include="object").columns.difference(
            [churn_col, "CustomerID", "customerid", "id"]
        ).tolist()

        if cat_cols:
            chosen_cat = st.selectbox("Break down churn by:", cat_cols, key="cat_sel")
            grp = (
                df.groupby(chosen_cat)["_churn"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "Churned", "count": "Total"})
                .assign(Rate=lambda x: x["Churned"] / x["Total"] * 100)
                .reset_index()
                .sort_values("Rate", ascending=False)
            )
            fig_bar = px.bar(
                grp, x=chosen_cat, y="Rate",
                color="Rate",
                color_continuous_scale=[[0, GREEN_LIGHT], [0.5, BLUE_MID], [1, DANGER]],
                labels={"Rate": "Churn Rate (%)"},
                text=grp["Rate"].apply(lambda v: f"{v:.1f}%"),
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                margin=dict(t=10, b=10, l=0, r=0),
                yaxis_title="Churn Rate (%)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            top_cat = grp.iloc[0]
            st.markdown(
                f"<div class='insight'>"
                f"<strong>{top_cat[chosen_cat]}</strong> has the highest churn rate at "
                f"<strong>{top_cat['Rate']:.1f}%</strong> "
                f"({int(top_cat['Churned']):,} of {int(top_cat['Total']):,} customers).</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No categorical columns detected for breakdown.")

    # Numeric distributions
    st.markdown("<div class='section-title'>Distribution by churn status</div>", unsafe_allow_html=True)
    if num_cols:
        chosen_num = st.selectbox("Numeric column:", num_cols, key="num_sel")
        fig_hist = px.histogram(
            df, x=chosen_num, color="_churn",
            nbins=40, barmode="overlay",
            color_discrete_map={0: GREEN_MID, 1: DANGER},
            labels={"_churn": "Churned", chosen_num: chosen_num},
            opacity=0.75,
        )
        fig_hist.for_each_trace(
            lambda t: t.update(name="Retained" if t.name == "0" else "Churned")
        )
        fig_hist = fig_layout(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEGMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_segments:
    st.markdown("<div class='section-title'>Churn rate across segments</div>", unsafe_allow_html=True)

    cat_cols2 = df.select_dtypes(include="object").columns.difference(
        [churn_col, "CustomerID", "customerid", "id"]
    ).tolist()

    if len(cat_cols2) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            row_var = st.selectbox("Rows (group by):", cat_cols2, key="row_var")
        with c2:
            remaining = [c for c in cat_cols2 if c != row_var]
            col_var   = st.selectbox("Columns (sub-group):", remaining, key="col_var") if remaining else None

        if col_var:
            pivot = (
                df.groupby([row_var, col_var])["_churn"]
                .mean()
                .mul(100)
                .reset_index()
                .pivot(index=row_var, columns=col_var, values="_churn")
                .fillna(0)
            )
            fig_heat = px.imshow(
                pivot,
                color_continuous_scale=[[0, GREEN_LIGHT], [0.5, BLUE_MID], [1, DANGER]],
                labels={"color": "Churn Rate (%)"},
                text_auto=".1f",
                aspect="auto",
            )
            fig_heat = fig_layout(fig_heat, "Churn rate heatmap (%)")
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Add a second categorical column for cross-segment heatmap.")

    # Box plots across categories
    if cat_cols2 and num_cols:
        st.markdown("<div class='section-title'>Numeric spread within segments</div>", unsafe_allow_html=True)
        bc1, bc2 = st.columns(2)
        with bc1:
            box_cat = st.selectbox("Category:", cat_cols2, key="box_cat")
        with bc2:
            box_num = st.selectbox("Metric:", num_cols, key="box_num")

        fig_box = px.box(
            df, x=box_cat, y=box_num, color="_churn",
            color_discrete_map={0: GREEN_MID, 1: DANGER},
            labels={"_churn": "Churned"},
        )
        fig_box.for_each_trace(
            lambda t: t.update(name="Retained" if t.name == "0" else "Churned")
        )
        fig_box = fig_layout(fig_box)
        st.plotly_chart(fig_box, use_container_width=True)

    # Correlation heatmap
    st.markdown("<div class='section-title'>Correlation with churn</div>", unsafe_allow_html=True)
    if num_cols:
        corr_df = df[num_cols + ["_churn"]].corr()[["_churn"]].drop("_churn")
        corr_df = corr_df.sort_values("_churn")
        fig_corr = px.bar(
            corr_df.reset_index(),
            x="index", y="_churn",
            color="_churn",
            color_continuous_scale=[[0, DANGER], [0.5, "#ffffff"], [1, GREEN_MID]],
            labels={"index": "Feature", "_churn": "Correlation with Churn"},
        )
        fig_corr.update_layout(
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        top_pos = corr_df["_churn"].idxmax()
        top_neg = corr_df["_churn"].idxmin()
        st.markdown(
            f"<div class='insight blue'>"
            f"<strong>{top_pos}</strong> is the strongest positive predictor of churn "
            f"(r = {corr_df.loc[top_pos,'_churn']:.2f}). "
            f"<strong>{top_neg}</strong> is the strongest negative predictor "
            f"(r = {corr_df.loc[top_neg,'_churn']:.2f}).</div>",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTION MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_model:
    if not run_model:
        st.info("Enable 'Run churn prediction model' in the sidebar to see this section.")
    else:
        st.markdown("<div class='section-title'>Random Forest churn classifier</div>", unsafe_allow_html=True)

        # Build feature matrix
        feature_df = df.drop(columns=[churn_col, "_churn"], errors="ignore")
        # Drop ID-like columns
        id_cols = [c for c in feature_df.columns
                   if "id" in c.lower() or c.lower() in {"customerid", "id", "customer"}]
        feature_df = feature_df.drop(columns=id_cols, errors="ignore")

        # Encode categoricals
        le_map = {}
        for col in feature_df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col].astype(str))
            le_map[col] = le

        feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

        if feature_df.shape[1] == 0:
            st.warning("No usable feature columns found after processing.")
        else:
            X = feature_df.values
            y = df["_churn"].values

            if len(np.unique(y)) < 2:
                st.warning("Only one class in the churn column — cannot train a classifier.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )

                with st.spinner("Training model…"):
                    clf = RandomForestClassifier(
                        n_estimators=150, max_depth=8,
                        class_weight="balanced", random_state=42, n_jobs=-1
                    )
                    clf.fit(X_train, y_train)

                y_pred  = clf.predict(X_test)
                y_proba = clf.predict_proba(X_test)[:, 1]
                auc     = roc_auc_score(y_test, y_proba)
                report  = classification_report(y_test, y_pred, output_dict=True)

                # ─ Model KPIs ─
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("ROC-AUC",   f"{auc:.3f}")
                m2.metric("Precision", f"{report['1']['precision']:.3f}")
                m3.metric("Recall",    f"{report['1']['recall']:.3f}")
                m4.metric("F1 Score",  f"{report['1']['f1-score']:.3f}")

                col_l, col_r = st.columns(2)

                # ─ Feature importances ─
                with col_l:
                    st.markdown(
                        "<div class='section-title'>Feature importance</div>",
                        unsafe_allow_html=True,
                    )
                    feat_imp = (
                        pd.Series(clf.feature_importances_, index=feature_df.columns)
                        .sort_values(ascending=True)
                        .tail(15)
                    )
                    fig_imp = go.Figure(go.Bar(
                        x=feat_imp.values,
                        y=feat_imp.index,
                        orientation="h",
                        marker_color=BLUE_MID,
                        marker_line_color=BLUE_DARK,
                        marker_line_width=0.5,
                    ))
                    fig_imp = fig_layout(fig_imp)
                    fig_imp.update_layout(xaxis_title="Importance", yaxis_title="")
                    st.plotly_chart(fig_imp, use_container_width=True)

                # ─ ROC curve ─
                with col_r:
                    st.markdown(
                        "<div class='section-title'>ROC curve</div>",
                        unsafe_allow_html=True,
                    )
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        line=dict(color=GREEN_MID, width=2.5),
                        name=f"AUC = {auc:.3f}",
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        line=dict(color="#aaa", dash="dash"), showlegend=False,
                    ))
                    fig_roc = fig_layout(fig_roc)
                    fig_roc.update_layout(
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

                # ─ Confusion matrix ─
                st.markdown(
                    "<div class='section-title'>Confusion matrix</div>",
                    unsafe_allow_html=True,
                )
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Retained", "Churned"],
                    y=["Retained", "Churned"],
                    color_continuous_scale=[[0, "#e8f5e9"], [1, GREEN_DARK]],
                    text_auto=True,
                )
                fig_cm = fig_layout(fig_cm)
                fig_cm.update_layout(coloraxis_showscale=False)
                cm_col, _ = st.columns([1, 1])
                with cm_col:
                    st.plotly_chart(fig_cm, use_container_width=True)

                # ─ Churn probability distribution ─
                st.markdown(
                    "<div class='section-title'>Predicted churn probability distribution</div>",
                    unsafe_allow_html=True,
                )
                prob_df = pd.DataFrame({"Probability": y_proba, "Actual": y_test})
                fig_prob = px.histogram(
                    prob_df, x="Probability", color="Actual", nbins=40,
                    barmode="overlay", opacity=0.75,
                    color_discrete_map={0: GREEN_MID, 1: DANGER},
                    labels={"Actual": "Churned"},
                )
                fig_prob.for_each_trace(
                    lambda t: t.update(name="Retained" if t.name == "0" else "Churned")
                )
                fig_prob = fig_layout(fig_prob)
                fig_prob.update_layout(xaxis_title="Churn Probability", yaxis_title="Count")
                st.plotly_chart(fig_prob, use_container_width=True)

                # ─ Download scored dataset ─
                st.markdown(
                    "<div class='section-title'>Download scored predictions</div>",
                    unsafe_allow_html=True,
                )
                all_proba = clf.predict_proba(X)[:, 1]
                scored = df_raw.copy()
                scored["Churn_Probability"] = all_proba.round(4)
                scored["Predicted_Churn"]   = (all_proba >= 0.5).astype(int)
                scored["Risk_Tier"] = pd.cut(
                    all_proba,
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=["Low", "Medium", "High"],
                )
                csv_bytes = scored.to_csv(index=False).encode()
                st.download_button(
                    label="Download scored CSV",
                    data=csv_bytes,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )

                # Risk tier breakdown
                tier_cnt = scored["Risk_Tier"].value_counts().reset_index()
                tier_cnt.columns = ["Risk Tier", "Count"]
                tier_cnt["Pct"] = (tier_cnt["Count"] / len(scored) * 100).round(1)
                st.dataframe(tier_cnt, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("<div class='section-title'>Dataset preview</div>", unsafe_allow_html=True)

    search = st.text_input("Filter rows (searches all text columns):", placeholder="e.g. Fiber optic")
    display_df = df_raw.copy()

    if search:
        mask = display_df.apply(
            lambda col: col.astype(str).str.contains(search, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]
        st.caption(f"{len(display_df):,} rows match '{search}'")

    st.dataframe(display_df, use_container_width=True, height=420)

    # Column summary
    st.markdown("<div class='section-title'>Column summary</div>", unsafe_allow_html=True)
    summary = pd.DataFrame({
        "Type":     df_raw.dtypes.astype(str),
        "Non-null": df_raw.notna().sum(),
        "Nulls":    df_raw.isna().sum(),
        "Unique":   df_raw.nunique(),
    })
    st.dataframe(summary, use_container_width=True)

    # Download filtered data
    dl_bytes = display_df.to_csv(index=False).encode()
    st.download_button(
        "Download filtered data as CSV",
        data=dl_bytes,
        file_name="filtered_churn_data.csv",
        mime="text/csv",
    )
