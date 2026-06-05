import json
import pickle
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning


BUNDLE_PATH = Path("models/gb_shap_inference_bundle.pkl")
DATA_PATH = Path("data/processed/icu_features.csv")
HOLDOUT_REPORT_PATH = Path("results/metrics/holdout_report.json")
GLOBAL_IMPORTANCE_PATH = Path("results/shap/shap_feature_importance.csv")
FIGURES_DIR = Path("results/figures")
GENDER_OPTIONS = {"Female": 0, "Male": 1}
FEATURE_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "heart_rate": "Heart Rate",
    "sbp": "SBP",
    "dbp": "DBP",
    "mbp": "MBP",
    "resp_rate": "Respiratory Rate",
    "spo2": "SpO2",
    "temperature": "Temperature",
}
PLOTLY_PAPER = "rgba(0,0,0,0)"
PLOTLY_PANEL = "#101a26"


st.set_page_config(
    page_title="ICU Mortality Risk Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 8% 12%, rgba(76, 147, 201, 0.18), transparent 26%),
                radial-gradient(circle at 92% 6%, rgba(224, 161, 76, 0.12), transparent 20%),
                linear-gradient(180deg, #07111a 0%, #0b1320 100%);
        }
        .block-container {
            max-width: 1680px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15, 25, 36, 0.98), rgba(11, 18, 27, 0.98));
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        .hero-shell {
            border-radius: 30px;
            border: 1px solid rgba(255,255,255,0.08);
            background:
                radial-gradient(circle at top left, rgba(71, 138, 196, 0.34), transparent 30%),
                radial-gradient(circle at 85% 20%, rgba(214, 161, 74, 0.14), transparent 18%),
                linear-gradient(135deg, rgba(13, 27, 42, 0.99), rgba(17, 29, 43, 0.96));
            box-shadow: 0 28px 72px rgba(0, 0, 0, 0.30);
            padding: 1.8rem 1.8rem 1.4rem 1.8rem;
            margin-bottom: 1.2rem;
        }
        .hero-kicker {
            font-size: 0.8rem;
            color: #9fc0db;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.45rem;
        }
        .hero-title {
            font-size: 2.8rem;
            line-height: 1.0;
            font-weight: 760;
            color: #f7fbff;
            margin-bottom: 0.6rem;
        }
        .hero-copy {
            color: #c3d4e4;
            max-width: 1120px;
            font-size: 1.05rem;
            line-height: 1.5;
        }
        .hero-row {
            display: flex;
            gap: 0.65rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        .hero-pill {
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.045);
            color: #ebf4fb;
            padding: 0.42rem 0.78rem;
            border-radius: 999px;
            font-size: 0.87rem;
        }
        .panel-title {
            color: #f3f8fd;
            font-size: 1.08rem;
            font-weight: 700;
            margin: 0.25rem 0 0.75rem 0;
        }
        .kpi-card {
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 22px;
            padding: 1.05rem 1.1rem 1rem 1.1rem;
            min-height: 132px;
            background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
        }
        .kpi-card.kpi-risk-high {
            background: linear-gradient(135deg, rgba(123, 31, 47, 0.96), rgba(84, 20, 32, 0.96));
        }
        .kpi-card.kpi-risk-low {
            background: linear-gradient(135deg, rgba(24, 88, 118, 0.96), rgba(17, 58, 78, 0.96));
        }
        .kpi-label {
            font-size: 0.78rem;
            color: #94afc6;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 2.1rem;
            font-weight: 760;
            color: #ffffff;
            line-height: 1.0;
        }
        .kpi-sub {
            font-size: 0.88rem;
            color: #d6e4ef;
            margin-top: 0.5rem;
            line-height: 1.35;
        }
        .callout {
            margin-top: 0.9rem;
            margin-bottom: 1rem;
            padding: 0.92rem 1rem;
            border-radius: 16px;
            border: 1px solid rgba(221, 169, 79, 0.15);
            background: linear-gradient(90deg, rgba(221, 169, 79, 0.12), rgba(221, 169, 79, 0.03));
            color: #f0dfb0;
            line-height: 1.4;
        }
        .side-note {
            color: #94afc6;
            font-size: 0.88rem;
            line-height: 1.45;
        }
        .sidebar-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.55rem;
            margin-bottom: 0.85rem;
        }
        .sidebar-card {
            border: 1px solid rgba(255,255,255,0.06);
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 0.75rem 0.7rem;
        }
        .sidebar-card-label {
            color: #88a4bc;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        }
        .sidebar-card-value {
            color: #f1f6fb;
            font-size: 1.15rem;
            font-weight: 720;
        }
        .signature {
            margin-top: 1.2rem;
            border-top: 1px solid rgba(255,255,255,0.06);
            padding-top: 0.95rem;
            color: #89a6bc;
            font-size: 0.86rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_plotly_layout(title: str, height: int = 330) -> dict:
    return {
        "title": {"text": title, "font": {"size": 20, "color": "#edf5fb"}},
        "paper_bgcolor": PLOTLY_PAPER,
        "plot_bgcolor": PLOTLY_PANEL,
        "font": {"color": "#dce8f3", "size": 14},
        "height": height,
        "margin": {"l": 28, "r": 24, "t": 60, "b": 38},
    }


@st.cache_resource
def load_bundle(path: Path) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        with path.open("rb") as file:
            return pickle.load(file)


@st.cache_data
def load_reference_data(path: Path, features: tuple[str, ...]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(features))


@st.cache_data
def load_holdout_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data
def load_global_importance(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def render_sidebar(bundle: dict, report: dict | None) -> None:
    with st.sidebar:
        st.markdown("### Model Snapshot")
        threshold = float(bundle["threshold"])
        roc_auc = f"{report['test_metrics']['roc_auc']:.3f}" if report else "n/a"
        mortality = f"{report['mortality_rate_test']:.1%}" if report else "n/a"
        st.markdown(
            f"""
            <div class="sidebar-grid">
                <div class="sidebar-card">
                    <div class="sidebar-card-label">Threshold</div>
                    <div class="sidebar-card-value">{threshold:.2f}</div>
                </div>
                <div class="sidebar-card">
                    <div class="sidebar-card-label">ROC-AUC</div>
                    <div class="sidebar-card-value">{roc_auc}</div>
                </div>
                <div class="sidebar-card">
                    <div class="sidebar-card-label">Mortality</div>
                    <div class="sidebar-card-value">{mortality}</div>
                </div>
                <div class="sidebar-card">
                    <div class="sidebar-card-label">Features</div>
                    <div class="sidebar-card-value">{len(bundle['features'])}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.write(f"Artifact: `{BUNDLE_PATH.name}`")
        st.write(f"Split mode: `{bundle.get('split_mode', 'n/a')}`")
        st.write(f"Seed: `{bundle.get('seed', 'n/a')}`")
        st.markdown("---")
        st.markdown(
            '<div class="side-note">Designed as a healthcare machine learning interface for interpretable risk assessment, '
            "model review, and case-based analysis. It is well suited to research exploration, "
            "demonstration workflows, and explainability-driven evaluation.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="signature">Workflow coverage includes data preprocessing, predictive modeling, '
            "holdout validation, SHAP-based explanation, and interactive patient-level scoring.</div>",
            unsafe_allow_html=True,
        )


def render_hero(bundle: dict, report: dict | None) -> None:
    roc_auc = f"{report['test_metrics']['roc_auc']:.3f}" if report else "n/a"
    pr_auc = f"{report['test_metrics']['pr_auc']:.3f}" if report else "n/a"
    threshold = float(bundle["threshold"])
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">Clinical Risk Intelligence</div>
            <div class="hero-title">ICU Mortality Risk Dashboard</div>
            <div class="hero-copy">
                Interactive inference and explanation interface for the saved gradient boosting model.
                It combines patient-level risk scoring, SHAP-based local interpretation, holdout performance,
                and cohort context to present model behavior in a clear, decision-oriented view.
            </div>
            <div class="hero-row">
                <div class="hero-pill">Threshold {threshold:.2f}</div>
                <div class="hero-pill">ROC-AUC {roc_auc}</div>
                <div class="hero-pill">PR-AUC {pr_auc}</div>
                <div class="hero-pill">Split {bundle.get("split_mode", "n/a")}</div>
                <div class="hero-pill">Explainable Gradient Boosting</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_manual_input(reference_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    defaults = reference_df.median(numeric_only=True).to_dict()
    st.markdown('<div class="panel-title">Manual Patient Entry</div>', unsafe_allow_html=True)
    left, right = st.columns([0.9, 2.3])

    with left:
        age = st.number_input("Age", min_value=0, max_value=120, value=int(defaults.get("age", 60)))
        gender_default = "Male" if int(defaults.get("gender", 0)) == 1 else "Female"
        gender_label = st.selectbox("Gender", options=list(GENDER_OPTIONS.keys()), index=list(GENDER_OPTIONS).index(gender_default))

    with right:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            heart_rate = st.number_input("Heart Rate", 0.0, 250.0, float(defaults.get("heart_rate", 90.0)))
            sbp = st.number_input("SBP", 0.0, 300.0, float(defaults.get("sbp", 120.0)))
        with c2:
            dbp = st.number_input("DBP", 0.0, 200.0, float(defaults.get("dbp", 70.0)))
            mbp = st.number_input("MBP", 0.0, 250.0, float(defaults.get("mbp", 85.0)))
        with c3:
            resp_rate = st.number_input("Respiratory Rate", 0.0, 80.0, float(defaults.get("resp_rate", 18.0)))
            spo2 = st.number_input("SpO2", 0.0, 100.0, float(defaults.get("spo2", 97.0)))
        with c4:
            temperature = st.number_input("Temperature", 80.0, 110.0, float(defaults.get("temperature", 98.6)))
            st.caption("Defaults come from processed-dataset medians.")

    row = {
        "age": age,
        "gender": GENDER_OPTIONS[gender_label],
        "heart_rate": heart_rate,
        "sbp": sbp,
        "dbp": dbp,
        "mbp": mbp,
        "resp_rate": resp_rate,
        "spo2": spo2,
        "temperature": temperature,
    }
    return pd.DataFrame([row])[features]


def build_sample_input(reference_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    st.markdown('<div class="panel-title">Reference Dataset Sample</div>', unsafe_allow_html=True)
    row_idx = st.slider("Select row from processed dataset", 0, len(reference_df) - 1, 0)
    sample = reference_df.iloc[[row_idx]].copy()
    st.dataframe(sample.rename(columns=FEATURE_LABELS), use_container_width=True, hide_index=True)
    return sample[features]


def get_prediction(bundle: dict, input_df: pd.DataFrame) -> tuple[float, int, pd.DataFrame]:
    features = bundle["features"]
    threshold = float(bundle["threshold"])
    imputed = bundle["imputer"].transform(input_df[features])
    imputed_df = pd.DataFrame(imputed, columns=features, index=input_df.index)
    probability = float(bundle["model"].predict_proba(imputed_df)[0, 1])
    prediction = int(probability >= threshold)
    return probability, prediction, imputed_df


def get_shap_contributions(bundle: dict, imputed_df: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = explainer.shap_values(imputed_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    values = shap_values[0] if getattr(shap_values, "ndim", 1) > 1 else shap_values
    contributions = pd.DataFrame(
        {
            "feature": imputed_df.columns,
            "value": imputed_df.iloc[0].values,
            "shap_contribution": values,
        }
    )
    contributions["abs_contribution"] = contributions["shap_contribution"].abs()
    contributions["direction"] = contributions["shap_contribution"].map(
        lambda value: "raises risk" if value > 0 else "lowers risk"
    )
    contributions["feature_label"] = contributions["feature"].map(FEATURE_LABELS)
    return contributions.sort_values("abs_contribution", ascending=False)


def render_kpi_card(label: str, value: str, sub: str, tone: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card {tone}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_gauge(probability: float, threshold: float) -> go.Figure:
    band_color = "#d2644a" if probability >= threshold else "#2e7aa7"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34, "color": "#f7fbff"}},
            title={"text": "Predicted Mortality Risk", "font": {"size": 18, "color": "#dce8f3"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#a6bfd3"},
                "bar": {"color": band_color, "thickness": 0.32},
                "bgcolor": "#132130",
                "bordercolor": "#24374b",
                "steps": [
                    {"range": [0, threshold * 100], "color": "#16384e"},
                    {"range": [threshold * 100, 100], "color": "#4d2028"},
                ],
                "threshold": {
                    "line": {"color": "#f1d48a", "width": 5},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(**build_plotly_layout("Risk Position", height=300))
    return fig


def local_contribution_chart(contributions: pd.DataFrame) -> go.Figure:
    top = contributions.head(8).sort_values("shap_contribution")
    colors = ["#d2644a" if value > 0 else "#3a83b7" for value in top["shap_contribution"]]
    fig = go.Figure(
        go.Bar(
            x=top["shap_contribution"],
            y=top["feature_label"],
            orientation="h",
            marker={"color": colors},
            text=[f"{value:+.3f}" for value in top["shap_contribution"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        **build_plotly_layout("Top Local Drivers", height=360),
        xaxis={"title": "SHAP contribution", "gridcolor": "#223446", "zerolinecolor": "#93a7b8"},
        yaxis={"title": "", "gridcolor": "rgba(0,0,0,0)"},
        showlegend=False,
    )
    return fig


def patient_vs_median_chart(input_df: pd.DataFrame, reference_df: pd.DataFrame) -> go.Figure:
    medians = reference_df.median(numeric_only=True)
    comp = pd.DataFrame(
        {
            "feature": input_df.columns,
            "patient": input_df.iloc[0].values,
            "median": medians[input_df.columns].values,
        }
    )
    comp = comp[comp["feature"] != "gender"].copy()
    comp["feature_label"] = comp["feature"].map(FEATURE_LABELS)
    comp["delta_pct"] = ((comp["patient"] - comp["median"]) / comp["median"].replace(0, pd.NA)) * 100
    comp = comp.sort_values("delta_pct")
    colors = ["#d2644a" if value > 0 else "#3a83b7" for value in comp["delta_pct"].fillna(0)]
    fig = go.Figure(
        go.Bar(
            x=comp["delta_pct"].fillna(0),
            y=comp["feature_label"],
            orientation="h",
            marker={"color": colors},
            text=[f"{value:+.1f}%" for value in comp["delta_pct"].fillna(0)],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        **build_plotly_layout("Patient vs Dataset Medians", height=360),
        xaxis={"title": "% difference from median", "gridcolor": "#223446", "zerolinecolor": "#93a7b8"},
        yaxis={"title": "", "gridcolor": "rgba(0,0,0,0)"},
        showlegend=False,
    )
    return fig


def global_importance_chart(global_importance: pd.DataFrame | None) -> go.Figure | None:
    if global_importance is None:
        return None
    data = global_importance.sort_values("mean_abs_shap")
    labels = [FEATURE_LABELS.get(feature, feature) for feature in data["feature"]]
    fig = go.Figure(
        go.Bar(
            x=data["mean_abs_shap"],
            y=labels,
            orientation="h",
            marker={"color": "#d5a14a"},
            text=[f"{value:.3f}" for value in data["mean_abs_shap"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        **build_plotly_layout("Global SHAP Importance", height=360),
        xaxis={"title": "Mean absolute SHAP", "gridcolor": "#223446"},
        yaxis={"title": "", "gridcolor": "rgba(0,0,0,0)"},
        showlegend=False,
    )
    return fig


def threshold_context_chart(probability: float, threshold: float, baseline: float | None) -> go.Figure:
    x = ["Baseline", "Threshold", "Current Patient"]
    y = [
        (baseline or 0) * 100,
        threshold * 100,
        probability * 100,
    ]
    colors = ["#6c8ca8", "#d5a14a", "#d2644a" if probability >= threshold else "#3a83b7"]
    fig = go.Figure(
        go.Bar(
            x=x,
            y=y,
            marker={"color": colors},
            text=[f"{value:.1f}%" for value in y],
            textposition="outside",
        )
    )
    fig.update_layout(
        **build_plotly_layout("Threshold Context", height=300),
        xaxis={"title": ""},
        yaxis={"title": "Probability (%)", "gridcolor": "#223446"},
        showlegend=False,
    )
    return fig


def render_prediction_summary(probability: float, prediction: int, threshold: float, report: dict | None) -> None:
    label = "High Risk" if prediction == 1 else "Lower Risk"
    tone = "kpi-risk-high" if prediction == 1 else "kpi-risk-low"
    baseline = report["mortality_rate_test"] if report else None
    c1, c2, c3 = st.columns(3)
    with c1:
        render_kpi_card(
            "Mortality Probability",
            f"{probability:.2%}",
            f"Compared with holdout baseline {baseline:.1%}" if baseline is not None else "Holdout baseline unavailable",
            tone,
        )
    with c2:
        render_kpi_card(
            "Decision Margin",
            f"{probability - threshold:+.2%}",
            f"Threshold set at {threshold:.2f}",
        )
    with c3:
        render_kpi_card(
            "Risk Label",
            label,
            "Crosses model threshold" if prediction == 1 else "Below model threshold",
        )

    message = (
        f"This patient is {'above' if prediction == 1 else 'below'} the model threshold of {threshold:.2f}. "
            "Use the explanation panels to see which features are contributing most strongly to that score."
        )
    st.markdown(f'<div class="callout">{message}</div>', unsafe_allow_html=True)


def render_model_performance(report: dict | None) -> None:
    if report is None:
        st.info("Holdout report not available.")
        return
    metrics = report["test_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    c3.metric("F1 @ Threshold", f"{metrics['f1_at_selected_threshold']:.3f}")
    c4.metric("Brier", f"{metrics['brier']:.3f}")

    ci = report.get("test_metric_ci95_bootstrap")
    if ci:
        st.caption(
            f"Bootstrap CI95: ROC-AUC {ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f}, "
            f"PR-AUC {ci['pr_auc'][0]:.3f}-{ci['pr_auc'][1]:.3f}"
        )


def render_figure_gallery() -> None:
    figure_paths = [
        FIGURES_DIR / "roc_curve.png",
        FIGURES_DIR / "threshold_sweep.png",
        FIGURES_DIR / "cv_calibration_comparison.png",
    ]
    available = [path for path in figure_paths if path.exists()]
    if not available:
        st.info("Evaluation figures are not available.")
        return
    cols = st.columns(len(available))
    for col, path in zip(cols, available):
        with col:
            st.image(str(path), caption=path.stem.replace("_", " ").title(), use_container_width=True)


def main() -> None:
    inject_styles()
    if not BUNDLE_PATH.exists():
        st.error(f"Missing model bundle: {BUNDLE_PATH}")
        return
    if not DATA_PATH.exists():
        st.error(f"Missing processed dataset: {DATA_PATH}")
        return

    bundle = load_bundle(BUNDLE_PATH)
    features = bundle["features"]
    threshold = float(bundle["threshold"])
    report = load_holdout_report(HOLDOUT_REPORT_PATH)
    global_importance = load_global_importance(GLOBAL_IMPORTANCE_PATH)
    reference_df = load_reference_data(DATA_PATH, tuple(features))

    render_sidebar(bundle, report)
    render_hero(bundle, report)

    mode = st.radio("Input Source", ["Manual entry", "Dataset sample"], horizontal=True)
    input_df = build_manual_input(reference_df, features) if mode == "Manual entry" else build_sample_input(reference_df, features)
    probability, prediction, imputed_df = get_prediction(bundle, input_df)
    contributions = get_shap_contributions(bundle, imputed_df)

    render_prediction_summary(probability, prediction, threshold, report)

    analysis_tab, performance_tab, details_tab = st.tabs(["Patient View", "Performance", "Model Details"])

    with analysis_tab:
        top_left, top_mid, top_right = st.columns([1.1, 1.1, 0.9])
        with top_left:
            st.plotly_chart(probability_gauge(probability, threshold), use_container_width=True)
        with top_mid:
            st.plotly_chart(
                threshold_context_chart(probability, threshold, report["mortality_rate_test"] if report else None),
                use_container_width=True,
            )
        with top_right:
            st.markdown('<div class="panel-title">Patient Snapshot</div>', unsafe_allow_html=True)
            st.dataframe(input_df.rename(columns=FEATURE_LABELS), use_container_width=True, hide_index=True)
            table_df = contributions[["feature_label", "value", "shap_contribution", "direction"]].rename(
                columns={"feature_label": "feature"}
            )
            st.markdown('<div class="panel-title">Top Contribution Table</div>', unsafe_allow_html=True)
            st.dataframe(table_df, use_container_width=True, hide_index=True, height=326)

        bottom_left, bottom_right = st.columns([1.25, 1.0])
        with bottom_left:
            st.plotly_chart(local_contribution_chart(contributions), use_container_width=True)
            st.plotly_chart(patient_vs_median_chart(input_df, reference_df), use_container_width=True)
        with bottom_right:
            st.markdown('<div class="panel-title">Global Feature Importance</div>', unsafe_allow_html=True)
            global_fig = global_importance_chart(global_importance)
            if global_fig is not None:
                st.plotly_chart(global_fig, use_container_width=True)
            else:
                st.info("Global SHAP importance file is not available.")

    with performance_tab:
        st.markdown('<div class="panel-title">Holdout Evaluation</div>', unsafe_allow_html=True)
        render_model_performance(report)
        st.markdown('<div class="panel-title">Evaluation Figures</div>', unsafe_allow_html=True)
        render_figure_gallery()

    with details_tab:
        st.markdown('<div class="panel-title">Bundle Metadata</div>', unsafe_allow_html=True)
        st.json(
            {
                "artifact": BUNDLE_PATH.name,
                "threshold": threshold,
                "features": features,
                "split_mode": bundle.get("split_mode", "n/a"),
                "seed": bundle.get("seed", "n/a"),
                "notes": bundle.get("notes", "n/a"),
            }
        )
        st.markdown('<div class="panel-title">Reference Medians</div>', unsafe_allow_html=True)
        medians = reference_df.median(numeric_only=True).rename("median").reset_index()
        medians["index"] = medians["index"].map(FEATURE_LABELS)
        st.dataframe(medians.rename(columns={"index": "feature"}), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
