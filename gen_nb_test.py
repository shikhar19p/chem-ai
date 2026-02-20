import json, os

base = r"C:\Users\shikhar pulluri\Desktop\murali projject"
nb_path = os.path.join(base, "notebooks", "reactive_extraction_pipeline.ipynb")

cells = []
cc = [0]

def md(src):
    cid = "md" + str(cc[0]); cc[0] += 1
    return {"cell_type": "markdown", "metadata": {}, "source": src, "id": cid}

def code(src):
    cid = "c" + str(cc[0]); cc[0] += 1
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src, "id": cid}


# ─── Cell 1: Title ────────────────────────────────────────────────────────────
cells.append(md([
    "# Reactive Extraction of Propionic Acid \u2014 ML/DL Prediction Pipeline\n",
    "## System: TBA (Tri-n-butylamine) / DES (Thymol:Menthol)\n",
    "\n",
    "This notebook trains and compares five predictive models:\n",
    "1. Polynomial Regression (RSM) + ANOVA\n",
    "2. Random Forest + SHAP\n",
    "3. XGBoost + SHAP\n",
    "4. Gaussian Process Regression + Bayesian Optimisation\n",
    "5. Artificial Neural Network (Keras)\n",
    "\n",
    "**Targets:** KD, E_pct, Z, SF_min  \n",
    "**Inputs:** Cin (N), TBA_pct (%), DES_ratio_num (1.0/1.5/2.0), + derived features"
]))

# ─── Cell 2: Setup ────────────────────────────────────────────────────────────
cells.append(code([
    "import subprocess, sys\n",
    "pkgs = ['xgboost', 'shap', 'statsmodels', 'bayesian-optimization', 'openpyxl']\n",
    "for pkg in pkgs:\n",
    "    try:\n",
    "        __import__(pkg.replace('-','_').split('[')[0])\n",
    "    except ImportError:\n",
    "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])\n",
    "\n",
    "import os, sys, warnings\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
    "if PROJECT_ROOT not in sys.path:\n",
    "    sys.path.insert(0, PROJECT_ROOT)\n",
    "\n",
    "from config import *\n",
    "from src.data_generator import load_or_generate, compute_nrtl_gamma\n",
    "from src.feature_engineering import add_polynomial_features, prepare_data, get_features\n",
    "from src.isotherm_fitting import fit_isotherms, plot_isotherms\n",
    "from src.metrics import compute_metrics, compile_metrics_table, print_metrics_table\n",
    "from src.plotting import (set_global_style, parity_plot, residual_plot,\n",
    "                           correlation_heatmap, comparison_heatmap, bar_comparison,\n",
    "                           learning_curve_ann)\n",
    "\n",
    "set_global_style()\n",
    "np.random.seed(RANDOM_SEED)\n",
    "print('Environment ready. Project root:', PROJECT_ROOT)"
]))

# ─── Cell 3: Section Header ───────────────────────────────────────────────────
cells.append(md(["# Section 1: Data Loading and Exploration"]))

# ─── Cell 4: Load Data ────────────────────────────────────────────────────────
cells.append(code([
    "df_raw, DATA_SOURCE = load_or_generate(RAW_DATA_CSV)\n",
    "print(f'Data source: {DATA_SOURCE}  |  Shape: {df_raw.shape}')\n",
    "df_raw.head(8)"
]))

# ─── Cell 5: EDA ──────────────────────────────────────────────────────────────
cells.append(code([
    "print('Descriptive statistics:')\n",
    "display(df_raw.describe().round(4))\n",
    "\n",
    "cols = ['Cin','TBA_pct','DES_ratio_num','KD','E_pct','Z','SF_min']\n",
    "correlation_heatmap(df_raw, columns=[c for c in cols if c in df_raw.columns],\n",
    "                    title='Feature & Target Correlation Heatmap',\n",
    "                    save_path=os.path.join(FIGURES_DIR,'eda_correlation.png'))\n",
    "\n",
    "fig, axes = plt.subplots(1,2, figsize=(12,4))\n",
    "for ax, target in zip(axes, ['KD','E_pct']):\n",
    "    for des in df_raw['DES_ratio_num'].unique():\n",
    "        sub = df_raw[df_raw['DES_ratio_num']==des].groupby('TBA_pct')[target].mean()\n",
    "        ax.plot(sub.index, sub.values, marker='o', label=f'DES={des}')\n",
    "    ax.set_xlabel('TBA (%)')\n",
    "    ax.set_ylabel(target)\n",
    "    ax.set_title(f'Mean {target} vs TBA% by DES ratio')\n",
    "    ax.legend(title='DES ratio')\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(FIGURES_DIR,'eda_trends.png'), dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ─── Cell 6: Feature Engineering ──────────────────────────────────────────────
cells.append(code([
    "df = add_polynomial_features(df_raw)\n",
    "print('Features after engineering:', [c for c in df.columns if c not in df_raw.columns])\n",
    "print('Total columns:', len(df.columns))\n",
    "df.head(3)"
]))

# ─── Cell 7: Section Header ───────────────────────────────────────────────────
cells.append(md(["# Section 2: Isotherm Fitting (Langmuir & Freundlich)"]))

# ─── Cell 8: Isotherms ────────────────────────────────────────────────────────
cells.append(code([
    "isotherm_rows = []\n",
    "for tba in sorted(df['TBA_pct'].unique()):\n",
    "    for des in sorted(df['DES_ratio_num'].unique()):\n",
    "        sub = df[(df['TBA_pct']==tba) & (df['DES_ratio_num']==des)]\n",
    "        if len(sub) < 3:\n",
    "            continue\n",
    "        Ce = sub['Caq_eq'].values\n",
    "        q  = sub['Corg_eq'].values\n",
    "        res = fit_isotherms(Ce, q)\n",
    "        row = {'TBA_pct': tba, 'DES_ratio': des}\n",
    "        if 'error' not in res.get('Langmuir',{'error':''}):\n",
    "            row.update({'Qmax': res['Langmuir']['Qmax'],\n",
    "                        'b':    res['Langmuir']['b'],\n",
    "                        'R2_L': res['Langmuir']['R2']})\n",
    "        if 'error' not in res.get('Freundlich',{'error':''}):\n",
    "            row.update({'Kf':   res['Freundlich']['Kf'],\n",
    "                        'n':    res['Freundlich']['n'],\n",
    "                        'R2_F': res['Freundlich']['R2']})\n",
    "        isotherm_rows.append(row)\n",
    "\n",
    "iso_df = pd.DataFrame(isotherm_rows)\n",
    "print('Isotherm parameters:')\n",
    "display(iso_df.round(4))\n",
    "\n",
    "sub = df[(df['TBA_pct']==20) & (df['DES_ratio_num']==1.0)]\n",
    "res = fit_isotherms(sub['Caq_eq'].values, sub['Corg_eq'].values)\n",
    "plot_isotherms(sub['Caq_eq'].values, sub['Corg_eq'].values, res,\n",
    "               title='Isotherm: TBA=20%, DES=1:1',\n",
    "               save_path=os.path.join(FIGURES_DIR,'isotherms_TBA20_DES1.png'))"
]))

# ─── Cell 9-10: RSM ───────────────────────────────────────────────────────────
cells.append(md(["# Section 3: Model 1 \u2014 Polynomial Regression (RSM) + ANOVA"]))

cells.append(code([
    "from src.models.regression import fit_rsm_anova\n",
    "\n",
    "results_dict = {}  # {model_name: {target: metrics}}\n",
    "rsm_results = {}\n",
    "for target in TARGETS:\n",
    "    if target not in df.columns:\n",
    "        continue\n",
    "    feat = get_features(df, 'base')\n",
    "    X_df = df[feat]\n",
    "    y    = df[target].values\n",
    "    model_rsm, anova_tbl, met, y_pred, feat_names = fit_rsm_anova(X_df, y, target)\n",
    "    rsm_results[target] = {'model': model_rsm, 'anova': anova_tbl,\n",
    "                            'metrics': met, 'y_pred': y_pred, 'y_true': y}\n",
    "    print(f'\n--- RSM: {target} ---')\n",
    "    print(f\"  R\u00b2={met['R2']:.4f}  Adj-R\u00b2={met['Adj_R2']:.4f}  RMSE={met['RMSE']:.4f}  MAE={met['MAE']:.4f}\")\n",
    "    print(f\"  F-stat={met['F_stat']:.2f}  p={met['F_pval']:.4e}\")\n",
    "    print('ANOVA Table:')\n",
    "    display(anova_tbl)\n",
    "\n",
    "results_dict['RSM'] = {t: rsm_results[t]['metrics'] for t in rsm_results}\n",
    "\n",
    "for target in TARGETS:\n",
    "    if target not in rsm_results:\n",
    "        continue\n",
    "    parity_plot(rsm_results[target]['y_true'], rsm_results[target]['y_pred'],\n",
    "                'RSM', target,\n",
    "                save_path=os.path.join(FIGURES_DIR, f'rsm_parity_{target}.png'))\n",
    "    residual_plot(rsm_results[target]['y_true'], rsm_results[target]['y_pred'],\n",
    "                  'RSM', target,\n",
    "                  save_path=os.path.join(FIGURES_DIR, f'rsm_residual_{target}.png'))"
]))

# ─── Cell 11-12: Random Forest ────────────────────────────────────────────────
cells.append(md(["# Section 4: Model 2 \u2014 Random Forest Regression + SHAP"]))

cells.append(code([
    "from src.models.random_forest import train_random_forest, plot_shap_summary, plot_rf_importance\n",
    "\n",
    "rf_results = {}\n",
    "for i, target in enumerate(TARGETS):\n",
    "    if target not in df.columns:\n",
    "        continue\n",
    "    tune = (i == 0)\n",
    "    _, _, _, _, X_tr, X_te, y_tr, y_te, _, _, feats = prepare_data(\n",
    "        df, target, feature_set='full', scale=False)\n",
    "    model_rf, met, shap_vals, explainer, y_pred = train_random_forest(\n",
    "        X_tr, X_te, y_tr, y_te, feats, tune=tune)\n",
    "    rf_results[target] = {'model': model_rf, 'metrics': met,\n",
    "                           'shap': shap_vals, 'y_pred': y_pred, 'y_true': y_te,\n",
    "                           'X_te': X_te, 'feats': feats}\n",
    "    print(f'RF {target}: R\u00b2={met[\"R2\"]:.4f}  RMSE={met[\"RMSE\"]:.4f}  CV_RMSE={met[\"CV_RMSE_mean\"]:.4f}\u00b1{met[\"CV_RMSE_std\"]:.4f}')\n",
    "\n",
    "results_dict['RandomForest'] = {t: rf_results[t]['metrics'] for t in rf_results}\n",
    "\n",
    "target = 'KD'\n",
    "parity_plot(rf_results[target]['y_true'], rf_results[target]['y_pred'],\n",
    "            'Random Forest', target,\n",
    "            save_path=os.path.join(FIGURES_DIR, f'rf_parity_{target}.png'))\n",
    "if rf_results[target]['shap'] is not None:\n",
    "    plot_shap_summary(rf_results[target]['shap'], rf_results[target]['X_te'],\n",
    "                      rf_results[target]['feats'], target,\n",
    "                      save_path=os.path.join(FIGURES_DIR, f'rf_shap_{target}.png'))\n",
    "plot_rf_importance(rf_results[target]['model'], rf_results[target]['feats'],\n",
    "                   target, save_path=os.path.join(FIGURES_DIR, f'rf_importance_{target}.png'))"
]))

# ─── Cell 13-14: XGBoost ──────────────────────────────────────────────────────
cells.append(md(["# Section 5: Model 3 \u2014 XGBoost Gradient Boosting + SHAP"]))

cells.append(code([
    "from src.models.xgboost_model import train_xgboost, plot_xgb_importance\n",
    "\n",
    "xgb_results = {}\n",
    "for target in TARGETS:\n",
    "    if target not in df.columns:\n",
    "        continue\n",
    "    _, _, _, _, X_tr, X_te, y_tr, y_te, _, _, feats = prepare_data(\n",
    "        df, target, feature_set='full', scale=False)\n",
    "    model_xgb, met, shap_vals, y_pred = train_xgboost(\n",
    "        X_tr, X_te, y_tr, y_te, feats)\n",
    "    xgb_results[target] = {'model': model_xgb, 'metrics': met,\n",
    "                            'shap': shap_vals, 'y_pred': y_pred,\n",
    "                            'y_true': y_te, 'X_te': X_te, 'feats': feats}\n",
    "    print(f'XGB {target}: R\u00b2={met[\"R2\"]:.4f}  RMSE={met[\"RMSE\"]:.4f}  CV_RMSE={met[\"CV_RMSE_mean\"]:.4f}')\n",
    "\n",
    "results_dict['XGBoost'] = {t: xgb_results[t]['metrics'] for t in xgb_results}\n",
    "\n",
    "target = 'KD'\n",
    "parity_plot(xgb_results[target]['y_true'], xgb_results[target]['y_pred'],\n",
    "            'XGBoost', target,\n",
    "            save_path=os.path.join(FIGURES_DIR, f'xgb_parity_{target}.png'))\n",
    "plot_xgb_importance(xgb_results[target]['model'], xgb_results[target]['feats'],\n",
    "                    target, save_path=os.path.join(FIGURES_DIR, f'xgb_importance_{target}.png'))\n",
    "if xgb_results[target]['shap'] is not None:\n",
    "    import shap\n",
    "    plt.figure(figsize=(9, 5))\n",
    "    shap.summary_plot(xgb_results[target]['shap'], xgb_results[target]['X_te'],\n",
    "                      feature_names=xgb_results[target]['feats'], show=False)\n",
    "    plt.title(f'XGBoost SHAP \u2014 {target}')\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(os.path.join(FIGURES_DIR, f'xgb_shap_{target}.png'), dpi=150, bbox_inches='tight')\n",
    "    plt.show()"
]))

# ─── Cell 15-17: GPR + Bayesian Opt ───────────────────────────────────────────
cells.append(md(["# Section 6: Model 4 \u2014 Gaussian Process Regression + Bayesian Optimisation"]))

cells.append(code([
    "from src.models.gpr_model import train_gpr, bayesian_optimization_suggest, plot_gpr_uncertainty\n",
    "\n",
    "gpr_results = {}\n",
    "for target in TARGETS:\n",
    "    if target not in df.columns:\n",
    "        continue\n",
    "    X_tr_s, X_te_s, y_tr_s, y_te_s, X_tr, X_te, y_tr, y_te, scX, scY, feats = prepare_data(\n",
    "        df, target, feature_set='base', scale=True)\n",
    "    gpr_mdl, y_pred, y_std, y_test_orig, met = train_gpr(X_tr_s, X_te_s, y_tr_s, y_te_s, scY)\n",
    "    gpr_results[target] = {'model': gpr_mdl, 'scaler_X': scX, 'scaler_y': scY,\n",
    "                            'metrics': met, 'y_pred': y_pred, 'y_std': y_std,\n",
    "                            'y_true': y_test_orig, 'feats': feats}\n",
    "    print(f'GPR {target}: R\u00b2={met[\"R2\"]:.4f}  RMSE={met[\"RMSE\"]:.4f}')\n",
    "    print(f'  Kernel: {met[\"kernel_fitted\"]}')\n",
    "\n",
    "results_dict['GPR'] = {t: gpr_results[t]['metrics'] for t in gpr_results}\n",
    "\n",
    "target = 'KD'\n",
    "parity_plot(gpr_results[target]['y_true'], gpr_results[target]['y_pred'],\n",
    "            'GPR', target, y_std=gpr_results[target]['y_std'],\n",
    "            save_path=os.path.join(FIGURES_DIR, f'gpr_parity_{target}.png'))\n",
    "plot_gpr_uncertainty(gpr_results[target]['y_true'], gpr_results[target]['y_pred'],\n",
    "                     gpr_results[target]['y_std'], target,\n",
    "                     save_path=os.path.join(FIGURES_DIR, f'gpr_uncertainty_{target}.png'))"
]))

cells.append(code([
    "target = 'KD'\n",
    "print('Running Bayesian Optimisation for next-experiment suggestions...')\n",
    "suggestions = bayesian_optimization_suggest(\n",
    "    gpr_results[target]['model'],\n",
    "    gpr_results[target]['scaler_X'],\n",
    "    gpr_results[target]['scaler_y'],\n",
    "    target_name=target\n",
    ")\n",
    "print(f'\nTop {len(suggestions)} suggested next experiments (maximise {target}):')\n",
    "display(suggestions)\n",
    "suggestions.to_csv(os.path.join(REPORTS_DIR, 'bayesian_suggestions.csv'), index=False)\n",
    "print('Saved to outputs/reports/bayesian_suggestions.csv')"
]))

# ─── Cell 18-20: ANN ──────────────────────────────────────────────────────────
cells.append(md(["# Section 7: Model 5 \u2014 Artificial Neural Network (Keras/TensorFlow)"]))

cells.append(code([
    "from src.models.ann_model import train_ann, save_ann\n",
    "\n",
    "ann_results = {}\n",
    "for target in TARGETS:\n",
    "    if target not in df.columns:\n",
    "        continue\n",
    "    X_tr_s, X_te_s, y_tr_s, y_te_s, _, _, _, _, _, scY, feats = prepare_data(\n",
    "        df, target, feature_set='full', scale=True)\n",
    "    model_ann, history, met, y_pred, y_test_orig = train_ann(\n",
    "        X_tr_s, X_te_s, y_tr_s, y_te_s, scY)\n",
    "    ann_results[target] = {'model': model_ann, 'history': history,\n",
    "                            'metrics': met, 'y_pred': y_pred, 'y_true': y_test_orig}\n",
    "    print(f'ANN {target}: R\u00b2={met[\"R2\"]:.4f}  RMSE={met[\"RMSE\"]:.4f}  Epochs={met[\"epochs_trained\"]}')\n",
    "    save_ann(model_ann, os.path.join(MODELS_DIR, f'ann_{target}.keras'))\n",
    "\n",
    "results_dict['ANN'] = {t: ann_results[t]['metrics'] for t in ann_results}"
]))

cells.append(code([
    "target = 'KD'\n",
    "learning_curve_ann(ann_results[target]['history'], target,\n",
    "                   save_path=os.path.join(FIGURES_DIR, f'ann_learning_{target}.png'))\n",
    "parity_plot(ann_results[target]['y_true'], ann_results[target]['y_pred'],\n",
    "            'ANN', target,\n",
    "            save_path=os.path.join(FIGURES_DIR, f'ann_parity_{target}.png'))"
]))

# ─── Cell 21-22: Comparison ───────────────────────────────────────────────────
cells.append(md(["# Section 8: Comparative Analysis \u2014 All Models \u00d7 All Targets"]))

cells.append(code([
    "metrics_df = compile_metrics_table(results_dict)\n",
    "metrics_df.to_csv(os.path.join(REPORTS_DIR, 'metrics_summary.csv'), index=False)\n",
    "print('Metrics saved to outputs/reports/metrics_summary.csv')\n",
    "\n",
    "print_metrics_table(metrics_df, 'R2')\n",
    "print_metrics_table(metrics_df, 'RMSE')\n",
    "\n",
    "comparison_heatmap(metrics_df, metric='R2',\n",
    "                   title='R\u00b2 Heatmap \u2014 All Models \u00d7 All Targets',\n",
    "                   save_path=os.path.join(FIGURES_DIR, 'comparison_R2_heatmap.png'))\n",
    "bar_comparison(metrics_df, metric='RMSE',\n",
    "               title='RMSE Comparison \u2014 All Models \u00d7 All Targets',\n",
    "               save_path=os.path.join(FIGURES_DIR, 'comparison_RMSE_bar.png'))\n",
    "\n",
    "best = metrics_df.loc[metrics_df.groupby('Target')['R2'].idxmax()][['Target','Model','R2','RMSE','MAE']]\n",
    "print('\nBest model per target (by R\u00b2):')\n",
    "display(best.reset_index(drop=True))"
]))

# ─── Cell 23-24: Save models ──────────────────────────────────────────────────
cells.append(md(["# Section 9: Save All Models and Summary"]))

cells.append(code([
    "import joblib\n",
    "\n",
    "for target in TARGETS:\n",
    "    if target in rsm_results:\n",
    "        joblib.dump(rsm_results[target]['model'],\n",
    "                    os.path.join(MODELS_DIR, f'rsm_{target}.pkl'))\n",
    "    if target in rf_results:\n",
    "        joblib.dump({'model': rf_results[target]['model']},\n",
    "                    os.path.join(MODELS_DIR, f'rf_{target}.pkl'))\n",
    "    if target in xgb_results:\n",
    "        joblib.dump(xgb_results[target]['model'],\n",
    "                    os.path.join(MODELS_DIR, f'xgb_{target}.pkl'))\n",
    "    if target in gpr_results:\n",
    "        joblib.dump({'model': gpr_results[target]['model'],\n",
    "                     'scaler_X': gpr_results[target]['scaler_X'],\n",
    "                     'scaler_y': gpr_results[target]['scaler_y']},\n",
    "                    os.path.join(MODELS_DIR, f'gpr_{target}.pkl'))\n",
    "\n",
    "print('All models saved to outputs/models/')\n",
    "print('All figures saved to outputs/figures/')\n",
    "print('Metrics CSV saved to outputs/reports/metrics_summary.csv')\n",
    "print('\nPipeline complete!')"
]))

# ─── Assemble and write notebook ──────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written: {nb_path}")
print(f"Total cells: {len(cells)}")
print(f"File size: {os.path.getsize(nb_path)} bytes")
