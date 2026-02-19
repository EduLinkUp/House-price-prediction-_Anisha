import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Fonts & base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Background ── */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* ── Cards ── */
    .card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.25), rgba(118,75,162,0.25));
        border: 1px solid rgba(102,126,234,0.40);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #a78bfa; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }

    /* ── Section headers ── */
    h1 { color: #f8fafc !important; font-weight: 700 !important; }
    h2 { color: #e2e8f0 !important; font-weight: 600 !important; }
    h3 { color: #cbd5e1 !important; font-weight: 500 !important; }

    /* ── Streamlit elements ── */
    .stSelectbox label, .stSlider label, .stNumberInput label { color: #cbd5e1 !important; }
    div[data-testid="stMetricValue"] { color: #a78bfa !important; font-size: 1.8rem !important; }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    div[data-testid="stMetricDelta"] svg { display: none; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(102,126,234,0.4);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8 !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* ── Success / info boxes ── */
    .stSuccess { background: rgba(16,185,129,0.15) !important; border-color: #10b981 !important; }
    .stInfo    { background: rgba(59,130,246,0.15) !important; border-color: #3b82f6 !important; }

    /* ── Prediction banner ── */
    .prediction-box {
        background: linear-gradient(135deg, #667eea33, #764ba233);
        border: 2px solid #667eea;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .prediction-price {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "CRIM":    "Per capita crime rate by town",
    "ZN":      "Proportion of residential land zoned for lots > 25,000 sq.ft.",
    "INDUS":   "Proportion of non-retail business acres per town",
    "CHAS":    "Charles River dummy variable (1 = bounds river, 0 = otherwise)",
    "NOX":     "Nitric oxides concentration (parts per 10 million)",
    "RM":      "Average number of rooms per dwelling",
    "AGE":     "Proportion of owner-occupied units built prior to 1940",
    "DIS":     "Weighted distances to five Boston employment centres",
    "RAD":     "Index of accessibility to radial highways",
    "TAX":     "Full-value property-tax rate per $10,000",
    "PTRATIO": "Pupil-teacher ratio by town",
    "B":       "1000(Bk − 0.63)² where Bk = proportion of Black residents",
    "LSTAT":   "Percentage of lower-status population",
}

PX_THEME = dict(
    template="plotly_dark",
)

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0", family="Inter"),
)


@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "HousingData.csv")
    return pd.read_csv(path)


@st.cache_resource
def train_models(df):
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    predictions = {
        "Linear Regression": lr.predict(X_test),
        "Random Forest":     rf.predict(X_test),
        "Ridge Regression":  ridge.predict(X_test),
    }

    cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring="r2")

    return {
        "models": {"Linear Regression": lr, "Random Forest": rf, "Ridge Regression": ridge},
        "imputer": imputer,
        "scaler": scaler,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "predictions": predictions,
        "feature_names": list(X.columns),
        "cv_scores": cv_scores,
        "X_scaled": X_scaled,
        "y": y,
    }


def compute_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return {"R² Score": r2, "RMSE": rmse, "MAE": mae}


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0;'>
        <div style='font-size:2.5rem;'>🏡</div>
        <div style='font-size:1.2rem; font-weight:700; color:#a78bfa;'>House Price</div>
        <div style='font-size:0.85rem; color:#64748b;'>Prediction Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Exploratory Analysis", "🤖 Model Performance", "🔮 Predict Price"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:#475569;'>Boston Housing Dataset<br>506 samples · 13 features</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────
df = load_data()
bundle = train_models(df)

# ═════════════════════════════════════════════
# PAGE: OVERVIEW
# ═════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🏡 House Price Prediction")
    st.markdown("<p style='color:#94a3b8; font-size:1.05rem;'>Boston Housing Dataset — Machine Learning Dashboard</p>", unsafe_allow_html=True)

    st.markdown("---")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(df)}</div>
            <div class='metric-label'>Total Samples</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{df.shape[1]-1}</div>
            <div class='metric-label'>Features</div></div>""", unsafe_allow_html=True)
    with col3:
        best_r2 = max(
            compute_metrics(bundle["y_test"], bundle["predictions"][m])["R² Score"]
            for m in bundle["predictions"]
        )
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{best_r2:.3f}</div>
            <div class='metric-label'>Best R² Score</div></div>""", unsafe_allow_html=True)
    with col4:
        missing = df.isnull().sum().sum()
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{missing}</div>
            <div class='metric-label'>Missing Values</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(
            df.head(10).style.background_gradient(cmap="Purples", axis=0),
            use_container_width=True,
            height=320,
        )

    with col_r:
        st.markdown("### 📈 Target Distribution")
        fig = px.histogram(
            df, x="MEDV", nbins=40, marginal="violin",
            color_discrete_sequence=["#667eea"],
            labels={"MEDV": "Median House Value ($1000s)"},
            **PX_THEME,
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📖 Feature Glossary")
    cols = st.columns(2)
    for i, (feat, desc) in enumerate(FEATURE_DESCRIPTIONS.items()):
        with cols[i % 2]:
            st.markdown(f"""<div class='card' style='padding:14px;'>
                <span style='color:#a78bfa; font-weight:600;'>{feat}</span>
                <span style='color:#94a3b8; font-size:0.85rem;'> — {desc}</span>
            </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════
# PAGE: EXPLORATORY ANALYSIS
# ═════════════════════════════════════════════
elif page == "📊 Exploratory Analysis":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Correlation", "Distributions", "Feature vs Target", "Missing Values"])

    # ── Tab 1: Heatmap ──
    with tab1:
        st.markdown("### Correlation Heatmap")
        corr = df.corr()
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            **PX_THEME,
        )
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=20, b=0))
        fig.update_traces(textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

        top = corr["MEDV"].drop("MEDV").abs().sort_values(ascending=False).head(5)
        st.markdown("**Top 5 features most correlated with house price:**")
        for feat, val in top.items():
            direction = "positive" if corr["MEDV"][feat] > 0 else "negative"
            color = "#10b981" if direction == "positive" else "#f43f5e"
            st.markdown(f"<span style='color:{color}'>{'▲' if direction=='positive' else '▼'} **{feat}** ({val:.3f} — {direction})</span>", unsafe_allow_html=True)

    # ── Tab 2: Distributions ──
    with tab2:
        st.markdown("### Feature Distributions")
        feature = st.selectbox("Select feature", [c for c in df.columns if c != "MEDV"])
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(
                df, x=feature, nbins=35, marginal="box",
                color_discrete_sequence=["#764ba2"],
                **PX_THEME,
            )
            fig.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0), title=f"Distribution of {feature}")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.box(
                df, y=feature,
                color_discrete_sequence=["#667eea"],
                **PX_THEME,
            )
            fig.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0), title=f"Box Plot — {feature}")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Basic statistics**")
        st.dataframe(df[[feature, "MEDV"]].describe().T.style.background_gradient(cmap="Purples"), use_container_width=True)

    # ── Tab 3: Feature vs MEDV ──
    with tab3:
        st.markdown("### Feature vs House Price (MEDV)")
        feat2 = st.selectbox("Select feature", [c for c in df.columns if c != "MEDV"], key="fvt")
        color_opt = st.checkbox("Color by CHAS (river boundary)", value=False)
        fig = px.scatter(
            df, x=feat2, y="MEDV",
            color="CHAS" if color_opt else None,
            trendline="ols",
            color_continuous_scale="Viridis",
            labels={"MEDV": "House Price ($1000s)", feat2: feat2},
            opacity=0.75,
            **PX_THEME,
        )
        fig.update_layout(height=450, margin=dict(l=0,r=0,t=20,b=0))
        fig.update_traces(marker=dict(size=7))
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Missing ──
    with tab4:
        st.markdown("### Missing Values Analysis")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Feature", "Missing Count"]
        missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
        missing_feat = missing[missing["Missing Count"] > 0]

        if missing_feat.empty:
            st.success("No missing values after imputation. All features are complete.")
        else:
            fig = px.bar(
                missing_feat.sort_values("Missing Count", ascending=True),
                x="Missing Count", y="Feature", orientation="h",
                color="Missing %", color_continuous_scale="Reds",
                **PX_THEME,
            )
            fig.update_layout(height=320, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(missing, use_container_width=True)

# ═════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("# 🤖 Model Performance")
    st.markdown("---")

    y_test = bundle["y_test"]
    preds  = bundle["predictions"]
    models = list(preds.keys())

    # ── Metrics table ──
    st.markdown("### Comparative Metrics")
    metrics_data = []
    for m in models:
        mm = compute_metrics(y_test, preds[m])
        metrics_data.append({"Model": m, **{k: round(v, 4) for k, v in mm.items()}})
    metrics_df = pd.DataFrame(metrics_data)

    cols = st.columns(len(models))
    for i, row in metrics_df.iterrows():
        with cols[i]:
            color = "#10b981" if row["R² Score"] == metrics_df["R² Score"].max() else "#a78bfa"
            st.markdown(f"""<div class='metric-card'>
                <div style='font-size:1rem; font-weight:600; color:{color}; margin-bottom:8px;'>{row['Model']}</div>
                <div style='font-size:1.6rem; font-weight:700; color:{color};'>{row['R² Score']:.4f}</div>
                <div style='color:#64748b; font-size:0.75rem;'>R² Score</div>
                <hr style='border-color:rgba(255,255,255,0.1); margin:10px 0;'>
                <div style='color:#94a3b8; font-size:0.85rem;'>RMSE: <b>{row['RMSE']:.3f}</b></div>
                <div style='color:#94a3b8; font-size:0.85rem;'>MAE: <b>{row['MAE']:.3f}</b></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    tab_a, tab_b, tab_c, tab_d = st.tabs(["Actual vs Predicted", "Residuals", "Feature Importance", "Cross Validation"])

    with tab_a:
        selected_model = st.selectbox("Select model", models)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=preds[selected_model],
            mode="markers",
            marker=dict(color="#667eea", size=8, opacity=0.75,
                        line=dict(width=1, color="rgba(255,255,255,0.2)")),
            name="Predictions",
        ))
        mn, mx = float(y_test.min()), float(y_test.max())
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines",
            line=dict(color="#f43f5e", width=2, dash="dash"),
            name="Perfect Prediction",
        ))
        fig.update_layout(
            xaxis_title="Actual Price ($1000s)",
            yaxis_title="Predicted Price ($1000s)",
            title=f"Actual vs Predicted — {selected_model}",
            height=450, **PLOTLY_THEME,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_b:
        selected_model_r = st.selectbox("Select model", models, key="res")
        residuals = np.array(y_test) - np.array(preds[selected_model_r])
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=preds[selected_model_r], y=residuals,
                mode="markers",
                marker=dict(color="#764ba2", size=7, opacity=0.7),
                name="Residuals",
            ))
            fig.add_hline(y=0, line=dict(color="#f43f5e", width=2, dash="dash"))
            fig.update_layout(
                xaxis_title="Predicted", yaxis_title="Residuals",
                title="Residual Plot", height=380, **PLOTLY_THEME,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col_r2:
            fig = px.histogram(
                x=residuals, nbins=30, marginal="violin",
                color_discrete_sequence=["#60a5fa"],
                labels={"x": "Residual"},
                **PX_THEME,
            )
            fig.update_layout(title="Residual Distribution", height=380, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with tab_c:
        rf_model = bundle["models"]["Random Forest"]
        importances = rf_model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": bundle["feature_names"],
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        fig = px.bar(
            feat_df, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=["#302b63", "#667eea", "#a78bfa"],
            labels={"Importance": "Importance Score"},
            **PX_THEME,
        )
        fig.update_layout(height=460, margin=dict(l=0,r=0,t=20,b=0),
                          title="Feature Importance — Random Forest")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_d:
        cv_scores = bundle["cv_scores"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(cv_scores))],
            y=cv_scores,
            marker=dict(
                color=cv_scores,
                colorscale=[[0, "#302b63"], [0.5, "#667eea"], [1, "#a78bfa"]],
                line=dict(width=1, color="rgba(255,255,255,0.15)"),
            ),
            text=[f"{v:.4f}" for v in cv_scores],
            textposition="outside",
        ))
        fig.add_hline(y=cv_scores.mean(), line=dict(color="#f43f5e", width=2, dash="dot"),
                      annotation_text=f"Mean = {cv_scores.mean():.4f}",
                      annotation_position="right")
        fig.update_layout(
            title="5-Fold Cross Validation R² — Random Forest",
            yaxis_title="R² Score", height=380, **PLOTLY_THEME,
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean R²",  f"{cv_scores.mean():.4f}")
        col2.metric("Std Dev",  f"{cv_scores.std():.4f}")
        col3.metric("Min → Max", f"{cv_scores.min():.3f} → {cv_scores.max():.3f}")

# ═════════════════════════════════════════════
# PAGE: PREDICT PRICE
# ═════════════════════════════════════════════
elif page == "🔮 Predict Price":
    st.markdown("# 🔮 Predict House Price")
    st.markdown("<p style='color:#94a3b8;'>Adjust the sliders to match your property profile and get an instant price estimate.</p>", unsafe_allow_html=True)
    st.markdown("---")

    feat_names = bundle["feature_names"]
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("### Property Parameters")
        input_vals = {}
        sliders_per_row = 2
        feat_cols = st.columns(sliders_per_row)
        for idx, feat in enumerate(feat_names):
            col = feat_cols[idx % sliders_per_row]
            with col:
                mn = float(df[feat].min())
                mx = float(df[feat].max())
                med = float(df[feat].median())
                # CHAS is binary
                if feat == "CHAS":
                    val = st.selectbox(
                        f"CHAS — Charles River boundary",
                        options=[0, 1],
                        format_func=lambda x: "Yes (bounds river)" if x == 1 else "No",
                        index=0,
                    )
                elif feat == "RAD":
                    val = st.slider(f"RAD — Highway access index", int(mn), int(mx), int(med), step=1)
                else:
                    val = st.slider(
                        f"{feat} — {FEATURE_DESCRIPTIONS.get(feat, feat)[:35]}",
                        min_value=round(mn, 2),
                        max_value=round(mx, 2),
                        value=round(med, 2),
                        step=round((mx - mn) / 100, 3) if mx != mn else 0.01,
                    )
                input_vals[feat] = val

    with col_b:
        st.markdown("### Select Model")
        chosen_model = st.selectbox("", list(bundle["models"].keys()), label_visibility="collapsed")

        st.markdown("---")

        if st.button("🔮 Predict Now"):
            input_arr = np.array([[input_vals[f] for f in feat_names]])
            imp = bundle["imputer"]
            scl = bundle["scaler"]
            mdl = bundle["models"][chosen_model]

            input_imputed = imp.transform(input_arr)
            input_scaled  = scl.transform(input_imputed)
            price = mdl.predict(input_scaled)[0]

            st.markdown(f"""
            <div class='prediction-box' style='margin-top:16px;'>
                <div style='color:#94a3b8; font-size:0.9rem;'>Estimated Price</div>
                <div class='prediction-price'>${price*1000:,.0f}</div>
                <div style='color:#64748b; font-size:0.8rem; margin-top:6px;'>{price:.2f} × $1,000</div>
                <div style='color:#a78bfa; font-size:0.85rem; margin-top:12px;'>Model: {chosen_model}</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=price,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Price ($1000s)", "font": {"color": "#e2e8f0", "size": 14}},
                number={"suffix": "K", "font": {"color": "#a78bfa", "size": 28}},
                gauge={
                    "axis": {"range": [0, 60], "tickcolor": "#64748b"},
                    "bar": {"color": "#667eea"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 20],  "color": "rgba(239,68,68,0.2)"},
                        {"range": [20, 40], "color": "rgba(234,179,8,0.2)"},
                        {"range": [40, 60], "color": "rgba(16,185,129,0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f43f5e", "width": 3},
                        "thickness": 0.75,
                        "value": price,
                    },
                },
            ))
            fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#e2e8f0", family="Inter"),
                              margin=dict(l=20,r=20,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

        # Quick comparison across models
        st.markdown("---")
        st.markdown("**Compare across all models**")
        input_arr = np.array([[input_vals[f] for f in feat_names]])
        input_imputed = bundle["imputer"].transform(input_arr)
        input_scaled  = bundle["scaler"].transform(input_imputed)

        comparison = {}
        for mname, mobj in bundle["models"].items():
            comparison[mname] = round(mobj.predict(input_scaled)[0], 2)

        for mname, pred in comparison.items():
            bar_pct = int(min(pred / 60 * 100, 100))
            color = "#667eea" if mname == chosen_model else "#475569"
            st.markdown(f"""
            <div style='margin-bottom:10px;'>
                <div style='display:flex; justify-content:space-between; color:#cbd5e1; font-size:0.85rem;'>
                    <span>{mname}</span><span><b>${pred}K</b></span>
                </div>
                <div style='background:rgba(255,255,255,0.06); border-radius:6px; height:8px; margin-top:4px;'>
                    <div style='background:{color}; width:{bar_pct}%; height:8px; border-radius:6px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
