"""
src/models/regression.py
Polynomial / RSM regression with full ANOVA table via statsmodels OLS.
"""

import numpy as np
import pandas as pd
import warnings
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.metrics import compute_metrics

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[regression] statsmodels not found — install with: pip install statsmodels")


def fit_rsm_anova(X_df, y, target_name='target', degree=2):
    """
    Fit a Response Surface (polynomial) regression model using OLS.
    Produces full ANOVA table with F-stats and p-values.

    Parameters
    ----------
    X_df : pd.DataFrame — predictor columns (already named)
    y    : array-like   — target values
    degree : int        — polynomial degree (default 2 for RSM)

    Returns
    -------
    model         : fitted statsmodels OLS result
    anova_table   : pd.DataFrame ANOVA table
    metrics       : dict {R2, RMSE, MAE, Adj_R2, F_stat, F_pval}
    y_pred        : predicted values (np.array)
    feature_names : list of feature column names used
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels is required. Run: pip install statsmodels")

    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_df.values)
    feature_names = poly.get_feature_names_out(X_df.columns.tolist()).tolist()

    # Sanitise column names: replace spaces/^ to avoid statsmodels parsing issues
    clean_names = []
    seen: dict = {}
    for n in feature_names:
        safe = n.replace(' ', '_').replace('^', 'pow')
        if safe in seen:
            seen[safe] += 1
            safe = f"{safe}_{seen[safe]}"
        else:
            seen[safe] = 0
        clean_names.append(safe)

    X_sm_df = pd.DataFrame(X_poly, columns=clean_names)
    X_sm_df.insert(0, 'const', 1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(np.asarray(y, dtype=float), X_sm_df).fit()

    y_pred = model.fittedvalues.values

    # Build coefficient significance table (more reliable than anova_lm on poly data)
    try:
        anova_table = pd.DataFrame({
            'coef':    model.params.values,
            'std_err': model.bse.values,
            't_stat':  model.tvalues.values,
            'p_value': model.pvalues.values,
        }, index=model.params.index)
        anova_table['significant'] = anova_table['p_value'] < 0.05
    except Exception as e2:
        anova_table = pd.DataFrame({'note': [f'ANOVA table error: {e2}']})

    metrics = {
        **compute_metrics(np.asarray(y), y_pred),
        'Adj_R2': float(model.rsquared_adj),
        'F_stat': float(model.fvalue) if hasattr(model, 'fvalue') else np.nan,
        'F_pval': float(model.f_pvalue) if hasattr(model, 'f_pvalue') else np.nan,
        'AIC': float(model.aic),
        'BIC': float(model.bic),
    }

    return model, anova_table, metrics, y_pred, feature_names


def predict_rsm(model, Xnew_df, degree=2, feature_names=None):
    """Run prediction on new DataFrame using a fitted RSM/OLS model."""
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(Xnew_df.values)
    col_names = ['const'] + poly.get_feature_names_out(Xnew_df.columns.tolist()).tolist()
    import statsmodels.api as sm
    X_sm_df = pd.DataFrame(sm.add_constant(X_poly), columns=col_names)
    return model.predict(X_sm_df)
