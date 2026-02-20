"""
run_pipeline.py
Command-line entry point for the reactive extraction ML pipeline.

Usage examples:
    python run_pipeline.py                              # synthetic data, all targets
    python run_pipeline.py --data data/raw/thesis_data.csv
    python run_pipeline.py --target KD
    python run_pipeline.py --data data/raw/thesis_data.csv --target KD E_pct
    python run_pipeline.py --models rsm rf xgb gpr ann
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import (TARGETS, RANDOM_SEED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
                    RAW_DATA_CSV)
from src.data_generator import load_or_generate
from src.feature_engineering import add_polynomial_features, prepare_data, get_features
from src.isotherm_fitting import fit_isotherms
from src.metrics import compute_metrics, compile_metrics_table, print_metrics_table
from src.plotting import (set_global_style, parity_plot, comparison_heatmap, bar_comparison)


# ── Helpers ───────────────────────────────────────────────────────────────────
def ensure_dirs():
    for d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)


def run_rsm(df, targets, results_dict):
    from src.models.regression import fit_rsm_anova
    for target in targets:
        feat = get_features(df, 'base')
        model, anova, met, y_pred, _ = fit_rsm_anova(df[feat], df[target].values, target)
        results_dict.setdefault('RSM', {})[target] = met
        joblib.dump(model, os.path.join(MODELS_DIR, f'rsm_{target}.pkl'))
        parity_plot(df[target].values, y_pred, 'RSM', target,
                    save_path=os.path.join(FIGURES_DIR, f'rsm_parity_{target}.png'))
        print(f'  RSM  {target:8s}: R²={met["R2"]:.4f}  RMSE={met["RMSE"]:.4f}')


def run_rf(df, targets, results_dict):
    from src.models.random_forest import train_random_forest, plot_rf_importance
    for i, target in enumerate(targets):
        _, _, _, _, X_tr, X_te, y_tr, y_te, _, _, feats = prepare_data(
            df, target, feature_set='full', scale=False)
        model, met, shap_vals, _, y_pred = train_random_forest(
            X_tr, X_te, y_tr, y_te, feats, tune=(i == 0))
        results_dict.setdefault('RandomForest', {})[target] = met
        joblib.dump(model, os.path.join(MODELS_DIR, f'rf_{target}.pkl'))
        parity_plot(y_te, y_pred, 'RF', target,
                    save_path=os.path.join(FIGURES_DIR, f'rf_parity_{target}.png'))
        print(f'  RF   {target:8s}: R²={met["R2"]:.4f}  RMSE={met["RMSE"]:.4f}')


def run_xgb(df, targets, results_dict):
    from src.models.xgboost_model import train_xgboost, plot_xgb_importance
    for target in targets:
        _, _, _, _, X_tr, X_te, y_tr, y_te, _, _, feats = prepare_data(
            df, target, feature_set='full', scale=False)
        model, met, _, y_pred = train_xgboost(X_tr, X_te, y_tr, y_te, feats)
        results_dict.setdefault('XGBoost', {})[target] = met
        joblib.dump(model, os.path.join(MODELS_DIR, f'xgb_{target}.pkl'))
        parity_plot(y_te, y_pred, 'XGBoost', target,
                    save_path=os.path.join(FIGURES_DIR, f'xgb_parity_{target}.png'))
        print(f'  XGB  {target:8s}: R²={met["R2"]:.4f}  RMSE={met["RMSE"]:.4f}')


def run_gpr(df, targets, results_dict):
    from src.models.gpr_model import (train_gpr, bayesian_optimization_suggest,
                                      plot_gpr_uncertainty)
    for target in targets:
        X_tr_s, X_te_s, y_tr_s, y_te_s, _, _, _, _, scX, scY, feats = prepare_data(
            df, target, feature_set='base', scale=True)
        gpr, y_pred, y_std, y_true, met = train_gpr(X_tr_s, X_te_s, y_tr_s, y_te_s, scY)
        results_dict.setdefault('GPR', {})[target] = met
        joblib.dump({'model': gpr, 'scaler_X': scX, 'scaler_y': scY},
                    os.path.join(MODELS_DIR, f'gpr_{target}.pkl'))
        parity_plot(y_true, y_pred, 'GPR', target, y_std=y_std,
                    save_path=os.path.join(FIGURES_DIR, f'gpr_parity_{target}.png'))
        print(f'  GPR  {target:8s}: R²={met["R2"]:.4f}  RMSE={met["RMSE"]:.4f}')

    # Bayesian Optimisation for first target only
    t = targets[0]
    pkg = joblib.load(os.path.join(MODELS_DIR, f'gpr_{t}.pkl'))
    sugg = bayesian_optimization_suggest(pkg['model'], pkg['scaler_X'],
                                         pkg['scaler_y'], target_name=t)
    if not sugg.empty:
        sugg.to_csv(os.path.join(REPORTS_DIR, 'bayesian_suggestions.csv'), index=False)
        print(f'\n  Bayesian Opt suggestions (target={t}):')
        print(sugg.to_string(index=False))


def run_ann(df, targets, results_dict):
    from src.models.ann_model import train_ann, save_ann
    for target in targets:
        X_tr_s, X_te_s, y_tr_s, y_te_s, _, _, _, _, _, scY, _ = prepare_data(
            df, target, feature_set='full', scale=True)
        model, _, met, y_pred, y_true = train_ann(X_tr_s, X_te_s, y_tr_s, y_te_s, scY)
        results_dict.setdefault('ANN', {})[target] = met
        save_ann(model, os.path.join(MODELS_DIR, f'ann_{target}.keras'))
        parity_plot(y_true, y_pred, 'ANN', target,
                    save_path=os.path.join(FIGURES_DIR, f'ann_parity_{target}.png'))
        print(f'  ANN  {target:8s}: R²={met["R2"]:.4f}  RMSE={met["RMSE"]:.4f}  '
              f'epochs={met["epochs_trained"]}')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Reactive Extraction ML Pipeline — Propionic Acid / TBA / DES'
    )
    parser.add_argument('--data',    default=RAW_DATA_CSV,
                        help='Path to CSV or "synthetic" (default: tries thesis_data.csv)')
    parser.add_argument('--target',  nargs='+', default=TARGETS,
                        help=f'Target(s) to predict (default: all — {TARGETS})')
    parser.add_argument('--models',  nargs='+',
                        default=['rsm', 'rf', 'xgb', 'gpr', 'ann'],
                        choices=['rsm', 'rf', 'xgb', 'gpr', 'ann'],
                        help='Models to run (default: all)')
    args = parser.parse_args()

    ensure_dirs()
    set_global_style()
    np.random.seed(RANDOM_SEED)

    # Load data
    csv_path = None if args.data == 'synthetic' else args.data
    df_raw, source = load_or_generate(csv_path or RAW_DATA_CSV)
    df = add_polynomial_features(df_raw)
    targets = [t for t in args.target if t in df.columns]
    if not targets:
        print(f"[ERROR] None of the requested targets {args.target} found in data.")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f'\n{"="*60}')
    print(f'  Reactive Extraction ML Pipeline')
    print(f'  Data source : {source}  |  Rows: {len(df)}')
    print(f'  Targets     : {targets}')
    print(f'  Models      : {args.models}')
    print(f'{"="*60}\n')

    results_dict = {}

    if 'rsm' in args.models:
        print('[1/5] Polynomial Regression + ANOVA ...')
        run_rsm(df, targets, results_dict)

    if 'rf' in args.models:
        print('\n[2/5] Random Forest ...')
        run_rf(df, targets, results_dict)

    if 'xgb' in args.models:
        print('\n[3/5] XGBoost ...')
        run_xgb(df, targets, results_dict)

    if 'gpr' in args.models:
        print('\n[4/5] Gaussian Process Regression + Bayesian Opt ...')
        run_gpr(df, targets, results_dict)

    if 'ann' in args.models:
        print('\n[5/5] Artificial Neural Network ...')
        run_ann(df, targets, results_dict)

    # Comparison
    if len(results_dict) > 1:
        metrics_df = compile_metrics_table(results_dict)
        metrics_df.to_csv(os.path.join(REPORTS_DIR, 'metrics_summary.csv'), index=False)
        print_metrics_table(metrics_df, 'R2')
        comparison_heatmap(metrics_df, metric='R2',
                           save_path=os.path.join(FIGURES_DIR, 'comparison_R2.png'))
        bar_comparison(metrics_df, metric='RMSE',
                       save_path=os.path.join(FIGURES_DIR, 'comparison_RMSE.png'))

    print(f'\nDone! Outputs saved to: {os.path.join(ROOT, "outputs")}')


if __name__ == '__main__':
    main()
