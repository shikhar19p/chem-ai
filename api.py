"""
api.py - FastAPI backend for Reactive Extraction Predictor
Run with:  uvicorn api:app --reload --port 8000
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

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (TARGETS, MODELS_DIR, REPORTS_DIR,
                    CIN_LEVELS, TBA_LEVELS, DES_RATIO_LEVELS)
from src.data_generator import compute_nrtl_gamma
from src.feature_engineering import build_single_input
from src.isotherm_fitting import fit_isotherms, langmuir_model, freundlich_model

app = FastAPI(title="Reactive Extraction Predictor API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once on startup ────────────────────────────────────────────────
_models = {}
_metrics_cache = []


def train_models_in_memory():
    """Train RF, XGBoost and GPR on synthetic data when no saved models exist."""
    print("[API] No saved models found — training on synthetic data...")
    from src.data_generator import generate_synthetic_dataset
    from src.feature_engineering import add_polynomial_features, prepare_data
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    from sklearn.preprocessing import PolynomialFeatures
    import xgboost as xgb
    import statsmodels.api as sm

    df = generate_synthetic_dataset(n_noise_runs=3)
    df = add_polynomial_features(df)

    trained = {}
    metrics = []

    for target in TARGETS:
        t_models = {}

        # ── Random Forest ──────────────────────────────────────────────────
        (X_tr_s, X_te_s, y_tr_s, y_te_s,
         X_tr, X_te, y_tr, y_te,
         scX, scY, feats) = prepare_data(df, target, feature_set='full', scale=False)

        rf = RandomForestRegressor(n_estimators=200, max_depth=7,
                                   random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        t_models['RandomForest'] = rf
        r2_rf = float(rf.score(X_te, y_te))
        metrics.append({'model': 'RandomForest', 'target': target, 'r2': round(r2_rf, 4)})

        # ── XGBoost ───────────────────────────────────────────────────────
        xgb_m = xgb.XGBRegressor(n_estimators=200, max_depth=4,
                                   learning_rate=0.05, subsample=0.8,
                                   random_state=42, verbosity=0)
        xgb_m.fit(X_tr, y_tr)
        t_models['XGBoost'] = xgb_m
        r2_xgb = float(xgb_m.score(X_te, y_te))
        metrics.append({'model': 'XGBoost', 'target': target, 'r2': round(r2_xgb, 4)})

        # ── GPR ───────────────────────────────────────────────────────────
        (X_tr_s2, X_te_s2, y_tr_s2, y_te_s2,
         _, _, _, _,
         scX2, scY2, feats_base) = prepare_data(df, target, feature_set='base', scale=True)

        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.01)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                        normalize_y=False, random_state=42)
        gpr.fit(X_tr_s2, y_tr_s2)
        r2_gpr = float(gpr.score(X_te_s2, y_te_s2))
        t_models['GPR'] = {'model': gpr, 'scaler_X': scX2, 'scaler_y': scY2}
        metrics.append({'model': 'GPR', 'target': target, 'r2': round(r2_gpr, 4)})

        # ── RSM (statsmodels OLS with polynomial features) ─────────────────
        try:
            (_, _, _, _,
             X_tr_b, X_te_b, y_tr_b, y_te_b,
             _, _, feats_b) = prepare_data(df, target, feature_set='base', scale=False)

            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X_tr_b)
            raw_names = poly.get_feature_names_out(feats_b).tolist()
            clean, seen = [], {}
            for n in raw_names:
                s = n.replace(' ', '_').replace('^', 'pow')
                if s in seen:
                    seen[s] += 1; s = f"{s}_{seen[s]}"
                else:
                    seen[s] = 0
                clean.append(s)
            X_sm = pd.DataFrame(X_poly, columns=clean)
            X_sm.insert(0, 'const', 1.0)
            rsm = sm.OLS(y_tr_b, X_sm).fit()
            t_models['RSM'] = rsm
            # score on test
            X_poly_te = poly.transform(X_te_b)
            X_sm_te = pd.DataFrame(X_poly_te, columns=clean)
            X_sm_te.insert(0, 'const', 1.0)
            y_pred_rsm = rsm.predict(X_sm_te)
            ss_res = np.sum((y_te_b - y_pred_rsm)**2)
            ss_tot = np.sum((y_te_b - np.mean(y_te_b))**2)
            r2_rsm = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            metrics.append({'model': 'RSM', 'target': target, 'r2': round(r2_rsm, 4)})
        except Exception as e:
            print(f"[API] RSM training failed for {target}: {e}")

        trained[target] = t_models
        print(f"[API] {target}: RF={r2_rf:.4f} XGB={r2_xgb:.4f} GPR={r2_gpr:.4f}")

    return trained, metrics


def load_all_models():
    global _models, _metrics_cache
    if _models:
        return _models

    # Try loading saved model files first
    any_found = False
    for target in TARGETS:
        t_models = {}
        p = os.path.join(MODELS_DIR, f'rsm_{target}.pkl')
        if os.path.exists(p):
            t_models['RSM'] = joblib.load(p); any_found = True
        p = os.path.join(MODELS_DIR, f'rf_{target}.pkl')
        if os.path.exists(p):
            t_models['RandomForest'] = joblib.load(p); any_found = True
        p = os.path.join(MODELS_DIR, f'xgb_{target}.pkl')
        if os.path.exists(p):
            t_models['XGBoost'] = joblib.load(p); any_found = True
        p = os.path.join(MODELS_DIR, f'gpr_{target}.pkl')
        if os.path.exists(p):
            t_models['GPR'] = joblib.load(p); any_found = True
        for ext in ['.keras', '.h5']:
            p = os.path.join(MODELS_DIR, f'ann_{target}{ext}')
            if os.path.exists(p):
                try:
                    from tensorflow import keras
                    t_models['ANN'] = keras.models.load_model(p)
                    any_found = True
                except Exception:
                    pass
                break
        _models[target] = t_models

    # If no saved models, train in memory on synthetic data
    if not any_found:
        _models, _metrics_cache = train_models_in_memory()

    return _models


@app.on_event("startup")
async def startup_event():
    import threading
    t = threading.Thread(target=load_all_models, daemon=True)
    t.start()
    print("[API] Startup complete — models loading in background.")


# ── Core prediction logic ──────────────────────────────────────────────────────
def predict_all(Cin, TBA_pct, DES_ratio_num):
    all_models = load_all_models()
    ga, go, C_TBA = compute_nrtl_gamma(Cin, TBA_pct, DES_ratio_num)

    X_full, feats_full = build_single_input(
        Cin, TBA_pct, DES_ratio_num, ga, go, C_TBA, feature_set='full')
    X_base, feats_base = build_single_input(
        Cin, TBA_pct, DES_ratio_num, ga, go, C_TBA, feature_set='base')

    preds = {}
    for target in TARGETS:
        t_preds = {}
        t_models = all_models.get(target, {})

        if 'RSM' in t_models:
            try:
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_df = pd.DataFrame(X_base, columns=feats_base)
                X_poly = poly.fit_transform(X_df.values)
                raw_names = poly.get_feature_names_out(feats_base).tolist()
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
                t_preds['RSM'] = round(val, 4)
            except Exception:
                pass

        if 'RandomForest' in t_models:
            try:
                m = t_models['RandomForest']
                if isinstance(m, dict): m = m['model']
                t_preds['RandomForest'] = round(float(m.predict(X_full)[0]), 4)
            except Exception:
                pass

        if 'XGBoost' in t_models:
            try:
                m = t_models['XGBoost']
                if isinstance(m, dict): m = m['model']
                t_preds['XGBoost'] = round(float(m.predict(X_full)[0]), 4)
            except Exception:
                pass

        if 'GPR' in t_models:
            try:
                pkg = t_models['GPR']
                gpr, scX, scY = pkg['model'], pkg['scaler_X'], pkg['scaler_y']
                X_s = scX.transform(X_base)
                mu, sg = gpr.predict(X_s, return_std=True)
                val = float(scY.inverse_transform(mu.reshape(-1, 1)).ravel()[0])
                std = float(sg[0]) * scY.scale_[0]
                t_preds['GPR'] = round(val, 4)
                t_preds['GPR_std'] = round(std, 4)
            except Exception:
                pass

        if 'ANN' in t_models:
            try:
                import joblib as _jl
                sc = _jl.load(os.path.join(MODELS_DIR, f'ann_scalers_{target}.pkl'))
                X_ann_s = sc['scaler_X'].transform(X_full)
                y_s = t_models['ANN'].predict(X_ann_s, verbose=0).ravel()
                val = float(sc['scaler_y'].inverse_transform(y_s.reshape(-1, 1)).ravel()[0])
                t_preds['ANN'] = round(val, 4)
            except Exception:
                pass

        preds[target] = t_preds

    return preds, ga, go, C_TBA


# ── Routes ─────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    Cin: float
    TBA_pct: float
    DES_ratio_num: float


@app.post("/predict")
def predict(req: PredictRequest):
    preds, ga, go, C_TBA = predict_all(req.Cin, req.TBA_pct, req.DES_ratio_num)

    in_range = (
        min(CIN_LEVELS) <= req.Cin <= max(CIN_LEVELS) and
        min(TBA_LEVELS) <= req.TBA_pct <= max(TBA_LEVELS) and
        req.DES_ratio_num in DES_RATIO_LEVELS
    )

    return {
        "predictions": preds,
        "nrtl": {
            "gamma_aq": round(ga, 4),
            "gamma_org": round(go, 4),
            "C_TBA_molar": round(C_TBA, 4),
        },
        "in_range": in_range,
    }


@app.get("/metrics")
def get_metrics():
    p = os.path.join(REPORTS_DIR, 'metrics_summary.csv')
    if os.path.exists(p):
        df = pd.read_csv(p)
        return df.to_dict(orient='records')
    # Fall back to in-memory metrics computed during startup training
    if _metrics_cache:
        return _metrics_cache
    return []


@app.get("/suggestions")
def get_suggestions():
    p = os.path.join(REPORTS_DIR, 'bayesian_suggestions.csv')
    if os.path.exists(p):
        df = pd.read_csv(p)
        return df.to_dict(orient='records')
    return []


@app.get("/sensitivity")
def get_sensitivity(
    xvar: str = Query("TBA_pct"),
    target: str = Query("KD"),
    steps: int = Query(10, ge=4, le=20),
    Cin: float = Query(0.10),
    TBA_pct: float = Query(10.0),
    DES_ratio_num: float = Query(1.5),
):
    """Sweep one input variable while holding others fixed."""
    x_ranges = {
        "Cin":           np.linspace(0.01, 1.0,  steps),
        "TBA_pct":       np.linspace(1,    100,   steps),
        "DES_ratio_num": np.linspace(0.5,  3.0,   steps),
    }
    if xvar not in x_ranges:
        return []

    results = []
    for val in x_ranges[xvar]:
        kwargs = {"Cin": Cin, "TBA_pct": TBA_pct, "DES_ratio_num": DES_ratio_num}
        kwargs[xvar] = float(val)
        preds, _, _, _ = predict_all(**kwargs)
        row = {xvar: round(float(val), 4)}
        for model in ['RSM', 'RandomForest', 'XGBoost', 'GPR', 'ANN']:
            v = preds.get(target, {}).get(model)
            row[model] = round(v, 4) if v is not None else None
        results.append(row)
    return results


@app.get("/matrix")
def get_matrix(
    target: str = Query("KD"),
    model: str = Query("GPR"),
    xvar: str = Query("TBA_pct"),
    yvar: str = Query("Cin"),
    steps: int = Query(8, ge=4, le=12),
    DES_ratio_num: float = Query(1.5),
    Cin: float = Query(0.10),
    TBA_pct: float = Query(10.0),
):
    """2-D grid heatmap: sweep xvar x yvar."""
    x_ranges = {
        "Cin":           np.linspace(0.01, 1.0,  steps),
        "TBA_pct":       np.linspace(1,    100,   steps),
        "DES_ratio_num": np.linspace(0.5,  3.0,   steps),
    }
    fixed = {"Cin": Cin, "TBA_pct": TBA_pct, "DES_ratio_num": DES_ratio_num}

    if xvar not in x_ranges or yvar not in x_ranges or xvar == yvar:
        return {"error": "xvar and yvar must be different valid variables"}

    x_vals = x_ranges[xvar]
    y_vals = x_ranges[yvar]

    grid = []
    for yv in y_vals:
        row = []
        for xv in x_vals:
            kwargs = dict(fixed)
            kwargs[xvar] = float(xv)
            kwargs[yvar] = float(yv)
            preds, _, _, _ = predict_all(**kwargs)
            v = preds.get(target, {}).get(model)
            row.append(round(v, 4) if v is not None else None)
        grid.append(row)

    return {
        "x_vals":  [round(v, 4) for v in x_vals],
        "y_vals":  [round(v, 4) for v in y_vals],
        "z_grid":  grid,
        "xvar":    xvar,
        "yvar":    yvar,
        "target":  target,
        "model":   model,
    }


@app.get("/isotherms")
def get_isotherms(
    TBA_pct: float = Query(10.0),
    DES_ratio_num: float = Query(1.5),
    n_points: int = Query(18, ge=5, le=40),
):
    """
    Compute adsorption isotherm: sweep Ce (0.01-0.8 N), predict Z via GPR,
    then fit Langmuir & Freundlich. Returns curve data + params.
    """
    Ce_vals = np.linspace(0.01, 0.80, n_points)
    q_vals  = []

    for ce in Ce_vals:
        preds, _, _, _ = predict_all(float(ce), TBA_pct, DES_ratio_num)
        z = preds.get('Z', {}).get('GPR')
        if z is None:
            z = preds.get('Z', {}).get('RandomForest', 0.0)
        q_vals.append(max(float(z), 0.0))

    Ce_arr = np.array(Ce_vals)
    q_arr  = np.array(q_vals)

    isofit = fit_isotherms(Ce_arr, q_arr)

    Ce_smooth = np.linspace(Ce_arr.min(), Ce_arr.max(), 120).tolist()
    lang_curve, frnd_curve = [], []

    lp = isofit.get('Langmuir', {})
    fp = isofit.get('Freundlich', {})

    if 'error' not in lp:
        lang_curve = [round(langmuir_model(c, lp['Qmax'], lp['b']), 6) for c in Ce_smooth]
    if 'error' not in fp:
        frnd_curve = [round(freundlich_model(c, fp['Kf'], fp['n']), 6) for c in Ce_smooth]

    return {
        "Ce":               [round(v, 4) for v in Ce_vals.tolist()],
        "q":                [round(v, 4) for v in q_vals],
        "Ce_smooth":        [round(v, 4) for v in Ce_smooth],
        "Langmuir_curve":   lang_curve,
        "Freundlich_curve": frnd_curve,
        "Langmuir_params":  {k: round(v, 4) for k, v in lp.items() if isinstance(v, float)},
        "Freundlich_params":{k: round(v, 4) for k, v in fp.items() if isinstance(v, float)},
    }


@app.get("/")
def root():
    return {"message": "Reactive Extraction Predictor API v2.0", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_models.keys())}
