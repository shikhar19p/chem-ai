"""
src/models/xgboost_model.py
XGBoost regression with SHAP feature importance analysis.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.metrics import compute_metrics
from config import XGB_PARAMS, CV_FOLDS, RANDOM_SEED

from sklearn.model_selection import cross_val_score, KFold


def train_xgboost(X_train, X_test, y_train, y_test,
                  feature_names,
                  params=None,
                  cv_folds=CV_FOLDS,
                  random_seed=RANDOM_SEED):
    """
    Train XGBoost regressor with 5-fold cross-validation.

    Parameters
    ----------
    X_train, X_test : unscaled arrays
    feature_names   : list of feature column names

    Returns
    -------
    model, metrics, shap_values, y_pred_test
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    model_params = XGB_PARAMS.copy() if params is None else params
    model_params['random_state'] = random_seed

    model = xgb.XGBRegressor(**model_params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=kf,
        scoring='neg_root_mean_squared_error'
    )

    metrics = {
        **compute_metrics(y_test, y_pred_test),
        'R2_train':    float(np.corrcoef(y_train, y_pred_train)[0, 1] ** 2),
        'CV_RMSE_mean': float(-cv_scores.mean()),
        'CV_RMSE_std':  float(cv_scores.std()),
    }

    # SHAP
    shap_values = None
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except ImportError:
        print("[XGB] shap not installed — skipping SHAP.")

    return model, metrics, shap_values, y_pred_test


def plot_xgb_importance(model, feature_names, target,
                         importance_type='gain', save_path=None):
    """Horizontal bar chart of XGBoost feature importances."""
    import matplotlib.pyplot as plt, os
    try:
        booster   = model.get_booster()
        score_dict = booster.get_score(importance_type=importance_type)
    except Exception:
        score_dict = dict(zip(feature_names, model.feature_importances_))

    # Map feature index names back to human-readable names
    # XGBoost uses f0, f1, ... internally
    renamed = {}
    for k, v in score_dict.items():
        if k.startswith('f') and k[1:].isdigit():
            idx = int(k[1:])
            renamed[feature_names[idx] if idx < len(feature_names) else k] = v
        else:
            renamed[k] = v

    if not renamed:
        print("[XGB] No feature importance scores available.")
        return

    sorted_items = sorted(renamed.items(), key=lambda x: x[1])
    names   = [x[0] for x in sorted_items]
    vals    = [x[1] for x in sorted_items]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.35)))
    ax.barh(names, vals, color='darkorange', edgecolor='saddlebrown')
    ax.set_xlabel(f'XGBoost Importance ({importance_type})')
    ax.set_title(f'XGBoost Feature Importance — {target}')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
