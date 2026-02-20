"""
src/models/random_forest.py
Random Forest regression with GridSearchCV hyperparameter tuning and SHAP analysis.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.metrics import compute_metrics
from config import RF_PARAM_GRID, CV_FOLDS, RANDOM_SEED

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV


def train_random_forest(X_train, X_test, y_train, y_test,
                         feature_names,
                         tune=True,
                         cv_folds=CV_FOLDS,
                         random_seed=RANDOM_SEED):
    """
    Train Random Forest with optional GridSearchCV tuning.

    Parameters
    ----------
    X_train, X_test : unscaled arrays (tree models don't need scaling)
    y_train, y_test : target arrays
    feature_names   : list of feature names
    tune            : bool — run GridSearchCV if True

    Returns
    -------
    model        : fitted RandomForestRegressor
    metrics      : dict
    shap_values  : np.array of SHAP values on X_test
    explainer    : shap.TreeExplainer
    y_pred_test  : predictions on X_test
    """
    if tune:
        gs = GridSearchCV(
            RandomForestRegressor(random_state=random_seed, n_jobs=-1),
            RF_PARAM_GRID,
            cv=cv_folds,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        best_params = gs.best_params_
    else:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=5,
            random_state=random_seed, n_jobs=-1
        )
        model.fit(X_train, y_train)
        best_params = model.get_params()

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv_folds,
        scoring='neg_root_mean_squared_error'
    )

    metrics = {
        **compute_metrics(y_test, y_pred_test),
        'R2_train': float(np.corrcoef(y_train, y_pred_train)[0, 1] ** 2),
        'CV_RMSE_mean': float(-cv_scores.mean()),
        'CV_RMSE_std':  float(cv_scores.std()),
        'best_params':  best_params,
    }

    # SHAP via TreeExplainer (exact, fast for RF)
    shap_values = None
    explainer   = None
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except ImportError:
        print("[RF] shap not installed — skipping SHAP. Run: pip install shap")

    return model, metrics, shap_values, explainer, y_pred_test


def plot_shap_summary(shap_values, X_test, feature_names, target, save_path=None):
    """SHAP beeswarm summary plot."""
    try:
        import shap, matplotlib.pyplot as plt, os
        plt.figure(figsize=(9, max(4, len(feature_names) * 0.4)))
        shap.summary_plot(shap_values, X_test,
                          feature_names=feature_names, show=False)
        plt.title(f'SHAP Feature Importance — {target}', fontsize=12)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    except ImportError:
        print("[RF] shap not installed — skipping plot.")


def plot_rf_importance(model, feature_names, target, save_path=None):
    """Simple bar chart of RF feature importances (MDI)."""
    import matplotlib.pyplot as plt, os
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, len(feature_names) * 0.35)))
    ax.barh([feature_names[i] for i in idx[::-1]],
            importances[idx[::-1]], color='steelblue', edgecolor='navy')
    ax.set_xlabel('MDI Importance')
    ax.set_title(f'Random Forest Feature Importance — {target}')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
