"""
src/metrics.py
Metric computation and reporting utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(y_true, y_pred, prefix=''):
    """Return dict with R2, RMSE, MAE (and optionally prefixed keys)."""
    r2   = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    if prefix:
        return {f'{prefix}R2': r2, f'{prefix}RMSE': rmse, f'{prefix}MAE': mae}
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae}


def compile_metrics_table(results_dict):
    """
    Build a DataFrame from a nested results dict.

    Expected format:
        results_dict[model_name][target_name] = {'R2': ..., 'RMSE': ..., 'MAE': ...}

    Returns multi-level DataFrame.
    """
    rows = []
    for model, targets in results_dict.items():
        for target, m in targets.items():
            rows.append({
                'Model': model, 'Target': target,
                'R2': m.get('R2', np.nan),
                'RMSE': m.get('RMSE', np.nan),
                'MAE': m.get('MAE', np.nan),
            })
    return pd.DataFrame(rows)


def print_metrics_table(df, metric='R2'):
    """Print a pivot table of one metric (model × target)."""
    pivot = df.pivot(index='Model', columns='Target', values=metric)
    print(f"\n{'='*60}")
    print(f"  {metric} — Model × Target")
    print('='*60)
    print(pivot.round(4).to_string())
    print('='*60)
    return pivot
