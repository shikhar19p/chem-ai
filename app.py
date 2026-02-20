"""
app.py — Streamlit Demo App for Reactive Extraction Predictor
Run with:  streamlit run app.py

Teacher/evaluator enters Cin, TBA%, DES ratio → app shows predictions
from all 5 trained models side-by-side.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

try:
    import streamlit as st
except ImportError:
    raise SystemExit("streamlit not installed. Run: pip install streamlit")

from config import (TARGETS, MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
                    CIN_LEVELS, TBA_LEVELS, DES_RATIO_LEVELS, RANDOM_SEED)
from src.data_generator import compute_nrtl_gamma, compute_all_targets, compute_kd
from src.feature_engineering import build_single_input

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reactive Extraction Predictor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load models (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading trained models...")
def load_all_models():
    """Load all saved model artefacts from outputs/models/."""
    models = {}
    for target in TARGETS:
        t_models = {}

        # RSM (statsmodels OLS)
        p = os.path.join(MODELS_DIR, f'rsm_{target}.pkl')
        if os.path.exists(p):
            t_models['RSM'] = joblib.load(p)

        # Random Forest
        p = os.path.join(MODELS_DIR, f'rf_{target}.pkl')
        if os.path.exists(p):
            t_models['RandomForest'] = joblib.load(p)

        # XGBoost
        p = os.path.join(MODELS_DIR, f'xgb_{target}.pkl')
        if os.path.exists(p):
            t_models['XGBoost'] = joblib.load(p)

        # GPR
        p = os.path.join(MODELS_DIR, f'gpr_{target}.pkl')
        if os.path.exists(p):
            t_models['GPR'] = joblib.load(p)

        # ANN (Keras)
        p = os.path.join(MODELS_DIR, f'ann_{target}.keras')
        if not os.path.exists(p):
            p = os.path.join(MODELS_DIR, f'ann_{target}.h5')
        if os.path.exists(p):
            try:
                from tensorflow import keras
                t_models['ANN'] = keras.models.load_model(p)
            except Exception:
                pass

        models[target] = t_models
    return models


@st.cache_data
def load_metrics():
    """Load saved metrics summary CSV."""
    p = os.path.join(REPORTS_DIR, 'metrics_summary.csv')
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data
def load_suggestions():
    p = os.path.join(REPORTS_DIR, 'bayesian_suggestions.csv')
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()


def predict_all_models(Cin, TBA_pct, DES_ratio_num, all_models):
    """
    Run prediction on given input using every loaded model.
    Returns dict: {target: {model_name: predicted_value}}
    """
    ga, go, C_TBA = compute_nrtl_gamma(Cin, TBA_pct, DES_ratio_num)

    # Build feature arrays for different feature sets
    X_full, feats_full = build_single_input(
        Cin, TBA_pct, DES_ratio_num, ga, go, C_TBA, feature_set='full')
    X_base, feats_base = build_single_input(
        Cin, TBA_pct, DES_ratio_num, ga, go, C_TBA, feature_set='base')

    preds = {}
    for target in TARGETS:
        t_preds = {}
        t_models = all_models.get(target, {})

        # RSM — must match column sanitisation used during training
        if 'RSM' in t_models:
            try:
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_df = pd.DataFrame(X_base, columns=feats_base)
                X_poly = poly.fit_transform(X_df.values)
                raw_names = poly.get_feature_names_out(feats_base).tolist()
                # Sanitise names exactly as done in regression.py
                clean_names, seen_n = [], {}
                for n in raw_names:
                    safe = n.replace(' ', '_').replace('^', 'pow')
                    if safe in seen_n:
                        seen_n[safe] += 1
                        safe = f"{safe}_{seen_n[safe]}"
                    else:
                        seen_n[safe] = 0
                    clean_names.append(safe)
                X_sm_df = pd.DataFrame(X_poly, columns=clean_names)
                X_sm_df.insert(0, 'const', 1.0)
                val = float(t_models['RSM'].predict(X_sm_df)[0])
                t_preds['RSM'] = val
            except Exception:
                pass

        # Random Forest
        if 'RandomForest' in t_models:
            try:
                m = t_models['RandomForest']
                # handle both plain model and dict
                if isinstance(m, dict):
                    m = m['model']
                t_preds['RandomForest'] = float(m.predict(X_full)[0])
            except Exception:
                pass

        # XGBoost
        if 'XGBoost' in t_models:
            try:
                m = t_models['XGBoost']
                if isinstance(m, dict):
                    m = m['model']
                t_preds['XGBoost'] = float(m.predict(X_full)[0])
            except Exception:
                pass

        # GPR
        if 'GPR' in t_models:
            try:
                pkg = t_models['GPR']
                gpr    = pkg['model']
                scX    = pkg['scaler_X']
                scY    = pkg['scaler_y']
                X_s    = scX.transform(X_base)
                mu, sg = gpr.predict(X_s, return_std=True)
                val    = float(scY.inverse_transform(mu.reshape(-1,1)).ravel()[0])
                std    = float(sg[0]) * scY.scale_[0]
                t_preds['GPR']         = val
                t_preds['GPR_std']     = std
            except Exception:
                pass

        # ANN — load saved scalers for proper scaling
        if 'ANN' in t_models:
            try:
                import joblib as _jl
                scaler_path = os.path.join(MODELS_DIR, f'ann_scalers_{target}.pkl')
                sc = _jl.load(scaler_path)
                X_ann_s = sc['scaler_X'].transform(X_full)
                y_s = t_models['ANN'].predict(X_ann_s, verbose=0).ravel()
                val = float(sc['scaler_y'].inverse_transform(y_s.reshape(-1, 1)).ravel()[0])
                t_preds['ANN'] = val
            except Exception:
                pass

        preds[target] = t_preds

    return preds


def format_pred_table(preds):
    """Build a DataFrame for display from preds dict."""
    model_names = ['RSM', 'RandomForest', 'XGBoost', 'GPR', 'ANN']
    rows = []
    for model in model_names:
        row = {'Model': model}
        for target in TARGETS:
            val = preds.get(target, {}).get(model, None)
            row[target] = round(val, 4) if val is not None else '—'
        rows.append(row)
    return pd.DataFrame(rows).set_index('Model')


# ─── Training check helper ───────────────────────────────────────────────────
def models_are_trained():
    return any(
        os.path.exists(os.path.join(MODELS_DIR, f'rf_{t}.pkl'))
        for t in TARGETS
    )


# ─── UI ───────────────────────────────────────────────────────────────────────
def main():
    st.title("⚗️ Reactive Extraction Predictor")
    st.markdown("**System:** Propionic Acid — TBA (Tri-n-butylamine) / DES (Thymol:Menthol)")
    st.markdown("Enter experimental conditions in the sidebar and press **Predict** to see predictions from all 5 ML models.")

    st.sidebar.header("🔬 Input Conditions")
    Cin = st.sidebar.slider("Initial Acid Concentration Cin (N)",
                             min_value=0.01, max_value=0.30,
                             value=0.10, step=0.01,
                             help="Initial propionic acid concentration (Normal)")
    TBA_pct = st.sidebar.slider("TBA Concentration (%)",
                                 min_value=1.0, max_value=30.0,
                                 value=10.0, step=0.5,
                                 help="Tri-n-butylamine wt% in organic phase")
    des_label = st.sidebar.selectbox("DES Molar Ratio (Thymol:Menthol)",
                                      options=["1:1 (1.0)", "1:1.5 (1.5)", "2:1 (2.0)"],
                                      index=0)
    DES_ratio_num = float(des_label.split('(')[1].rstrip(')'))

    # Show range warning
    in_range = (
        min(CIN_LEVELS) <= Cin <= max(CIN_LEVELS) and
        min(TBA_LEVELS) <= TBA_pct <= max(TBA_LEVELS) and
        DES_ratio_num in DES_RATIO_LEVELS
    )
    if not in_range:
        st.sidebar.warning("⚠️ Input is outside the training data range — extrapolation mode")
    else:
        st.sidebar.success("✅ Input is within training range")

    predict_btn = st.sidebar.button("🔮 Predict", type="primary", use_container_width=True)

    # ── NRTL reference values ────────────────────────────────────────────────
    ga, go, C_TBA = compute_nrtl_gamma(Cin, TBA_pct, DES_ratio_num)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**NRTL Estimated Values**")
    st.sidebar.metric("γ_aq", f"{ga:.4f}")
    st.sidebar.metric("γ_org", f"{go:.4f}")
    st.sidebar.metric("C_TBA (mol/L)", f"{C_TBA:.4f}")

    # ── Main area tabs ────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📈 Model Metrics", "🔬 Next Experiments"])

    with tab1:
        if not models_are_trained():
            st.warning(
                "No trained models found in `outputs/models/`. "
                "Please run the training pipeline first:\n\n"
                "```\npython run_pipeline.py\n```\n\n"
                "or open and run the Jupyter notebook `notebooks/reactive_extraction_pipeline.ipynb`."
            )
            st.stop()

        if predict_btn:
            all_models = load_all_models()

            with st.spinner("Running all models..."):
                preds = predict_all_models(Cin, TBA_pct, DES_ratio_num, all_models)

            pred_df = format_pred_table(preds)
            st.subheader("Prediction Results — All 5 Models")

            # Colour the table: green = high R², gradient
            st.dataframe(
                pred_df.style.format({t: '{:.4f}' for t in TARGETS
                                      if t in pred_df.columns}, na_rep='—'),
                use_container_width=True
            )

            # Bar charts side by side
            col1, col2 = st.columns(2)
            for col, target in zip([col1, col2], ['KD', 'E_pct']):
                with col:
                    vals = {
                        m: preds[target].get(m)
                        for m in ['RSM','RandomForest','XGBoost','GPR','ANN']
                        if preds[target].get(m) is not None
                    }
                    if vals:
                        chart_df = pd.DataFrame.from_dict(
                            {'Model': list(vals.keys()), target: list(vals.values())}
                        ).set_index('Model')
                        st.markdown(f"**{target} by Model**")
                        st.bar_chart(chart_df)

            # GPR uncertainty
            gpr_kd = preds.get('KD', {})
            if 'GPR' in gpr_kd and 'GPR_std' in gpr_kd:
                st.info(
                    f"GPR Uncertainty (KD): {gpr_kd['GPR']:.4f} ± "
                    f"{2*gpr_kd['GPR_std']:.4f}  (95% CI)"
                )
        else:
            st.info("👈 Set conditions in the sidebar and click **Predict**.")

    with tab2:
        metrics_df = load_metrics()
        if metrics_df.empty:
            st.warning("No metrics found. Run training first.")
        else:
            st.subheader("Model Performance — Training Results")
            for metric in ['R2', 'RMSE', 'MAE']:
                try:
                    pivot = metrics_df.pivot(index='Model', columns='Target', values=metric)
                    st.markdown(f"**{metric}**")
                    st.dataframe(pivot.style.format('{:.4f}'), use_container_width=True)
                except Exception:
                    pass

    with tab3:
        sugg_df = load_suggestions()
        if sugg_df.empty:
            st.warning("No Bayesian Optimisation suggestions found yet. Run training first.")
        else:
            st.subheader("🔬 Bayesian Optimisation: Suggested Next Experiments")
            st.markdown(
                "These 5 experimental conditions are predicted to maximise **KD** "
                "while also exploring uncertain regions (Upper Confidence Bound criterion)."
            )
            st.dataframe(sugg_df.style.format('{:.4f}'), use_container_width=True)
            st.download_button(
                "📥 Download suggestions CSV",
                data=sugg_df.to_csv(index=False),
                file_name='bayesian_suggestions.csv',
                mime='text/csv'
            )


if __name__ == '__main__':
    main()
