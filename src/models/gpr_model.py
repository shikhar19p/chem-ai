"""
src/models/gpr_model.py
Gaussian Process Regression with Matern kernel + Bayesian Optimisation
for suggesting next experimental conditions.
"""

import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.metrics import compute_metrics
from config import (
    GPR_N_RESTARTS, RANDOM_SEED,
    BAYES_N_INIT, BAYES_N_ITER, BAYES_NEXT_N, BAYES_PBOUNDS
)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, ConstantKernel, WhiteKernel
)


def train_gpr(X_train_s, X_test_s, y_train_s, y_test_s,
              scaler_y,
              n_restarts=GPR_N_RESTARTS,
              random_seed=RANDOM_SEED):
    """
    Train GPR with composite Matern(2.5) kernel.

    Parameters
    ----------
    *_s      : StandardScaler-transformed arrays
    scaler_y : fitted scaler used to inverse-transform predictions

    Returns
    -------
    gpr          : fitted GaussianProcessRegressor
    y_pred_orig  : predictions in original scale
    y_std_orig   : std dev in original scale
    y_test_orig  : true values in original scale
    metrics      : dict
    """
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3)) *
        Matern(length_scale=1.0, length_scale_bounds=(1e-2, 10.0), nu=2.5) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        random_state=random_seed,
        alpha=1e-6,
    )
    gpr.fit(X_train_s, y_train_s)

    y_pred_s, y_std_s = gpr.predict(X_test_s, return_std=True)

    # Inverse-transform to original units
    y_pred_orig = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_std_orig  = y_std_s * scaler_y.scale_[0]
    y_test_orig = scaler_y.inverse_transform(y_test_s.reshape(-1, 1)).ravel()

    metrics = {
        **compute_metrics(y_test_orig, y_pred_orig),
        'log_marginal_likelihood': float(gpr.log_marginal_likelihood_value_),
        'kernel_fitted': str(gpr.kernel_),
    }

    return gpr, y_pred_orig, y_std_orig, y_test_orig, metrics


def bayesian_optimization_suggest(gpr, scaler_X, scaler_y,
                                   target_name='KD',
                                   n_init=BAYES_N_INIT,
                                   n_iter=BAYES_N_ITER,
                                   next_n=BAYES_NEXT_N,
                                   pbounds=None,
                                   random_seed=RANDOM_SEED):
    """
    Use UCB Bayesian Optimisation to suggest next experiments.

    The surrogate (GPR) is queried with UCB = mu + 2.576*sigma.

    Returns
    -------
    pd.DataFrame  — top `next_n` suggested conditions
    """
    if pbounds is None:
        pbounds = BAYES_PBOUNDS

    try:
        from bayes_opt import BayesianOptimization
    except ImportError:
        print("[GPR] bayesian-optimization not installed. "
              "Run: pip install bayesian-optimization")
        return pd.DataFrame()

    def surrogate(Cin, TBA_pct, DES_ratio_num):
        x = np.array([[Cin, TBA_pct, DES_ratio_num]])
        x_s = scaler_X.transform(x)
        mu, sigma = gpr.predict(x_s, return_std=True)
        # UCB: kappa = 2.576 → 99% confidence upper bound
        return float(mu[0] + 2.576 * sigma[0])

    optimizer = BayesianOptimization(
        f=surrogate,
        pbounds=pbounds,
        random_state=random_seed,
        verbose=0,
    )
    optimizer.maximize(init_points=n_init, n_iter=n_iter)

    # Collect unique top-N suggestions
    results_sorted = sorted(optimizer.res, key=lambda r: -r['target'])
    suggestions = []
    seen = set()
    for res in results_sorted:
        key = (round(res['params']['Cin'], 3),
               round(res['params']['TBA_pct'], 1),
               round(res['params']['DES_ratio_num'], 2))
        if key not in seen:
            seen.add(key)
            # Back-transform UCB to approximate original-scale prediction
            x_s = scaler_X.transform([[
                res['params']['Cin'],
                res['params']['TBA_pct'],
                res['params']['DES_ratio_num']
            ]])
            mu_s, sig_s = gpr.predict(x_s, return_std=True)
            mu_orig  = scaler_y.inverse_transform(mu_s.reshape(-1, 1)).ravel()[0]
            std_orig = float(sig_s[0]) * scaler_y.scale_[0]
            suggestions.append({
                'Cin (N)':            round(res['params']['Cin'], 4),
                'TBA_pct (%)':        round(res['params']['TBA_pct'], 2),
                'DES_ratio_num':      round(res['params']['DES_ratio_num'], 3),
                f'Pred_{target_name}': round(mu_orig, 4),
                'Uncertainty (±2σ)':  round(2 * std_orig, 4),
            })
        if len(suggestions) >= next_n:
            break

    return pd.DataFrame(suggestions)


def plot_gpr_uncertainty(y_test_orig, y_pred_orig, y_std_orig,
                          target, save_path=None):
    """Sorted prediction plot with ±2σ uncertainty band."""
    import matplotlib.pyplot as plt, os
    idx = np.argsort(y_test_orig)
    yt  = y_test_orig[idx]
    yp  = y_pred_orig[idx]
    ys  = y_std_orig[idx]

    fig, ax = plt.subplots(figsize=(9, 4))
    x_ax = np.arange(len(yt))
    ax.fill_between(x_ax, yp - 2*ys, yp + 2*ys,
                    alpha=0.25, color='steelblue', label='±2σ band')
    ax.plot(x_ax, yp, 'b-', lw=1.5, label='GPR Prediction')
    ax.scatter(x_ax, yt, color='red', s=40, zorder=5, label='Actual')
    ax.set_xlabel('Sample index (sorted by actual value)')
    ax.set_ylabel(target)
    ax.set_title(f'GPR Uncertainty Quantification — {target}')
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
