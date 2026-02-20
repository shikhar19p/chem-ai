"""
src/isotherm_fitting.py
Langmuir and Freundlich isotherm fitting using scipy.optimize.curve_fit.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
import os


def langmuir_model(Ce, Qmax, b):
    """Langmuir isotherm: q = (Qmax * b * Ce) / (1 + b * Ce)"""
    return (Qmax * b * Ce) / (1.0 + b * Ce)


def freundlich_model(Ce, Kf, n):
    """Freundlich isotherm: q = Kf * Ce^(1/n)"""
    return Kf * np.power(np.clip(Ce, 1e-12, None), 1.0 / n)


def r2_score_manual(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_isotherms(Ce_array, q_array):
    """
    Fit Langmuir and Freundlich isotherms to equilibrium data.

    Parameters
    ----------
    Ce_array : array-like  — equilibrium aqueous concentration (N)
    q_array  : array-like  — loading (Corg_eq or Z)

    Returns
    -------
    dict with keys 'Langmuir' and 'Freundlich', each containing:
        params (fitted), R2, RMSE, and either (Qmax, b) or (Kf, n).
        On failure, contains 'error' key.
    """
    Ce = np.asarray(Ce_array, dtype=float)
    q  = np.asarray(q_array,  dtype=float)

    # Remove non-positive values that would break log transforms
    mask = (Ce > 0) & (q > 0)
    Ce, q = Ce[mask], q[mask]

    results = {}

    # ── Langmuir ──────────────────────────────────────────────────────────────
    try:
        p0 = [max(q) * 1.2, 1.0 / np.mean(Ce)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                langmuir_model, Ce, q,
                p0=p0,
                bounds=([0.0, 0.0], [np.inf, np.inf]),
                maxfev=20000,
            )
        q_pred = langmuir_model(Ce, *popt)
        r2   = r2_score_manual(q, q_pred)
        rmse = np.sqrt(np.mean((q - q_pred) ** 2))
        results['Langmuir'] = {
            'Qmax': float(popt[0]), 'b': float(popt[1]),
            'R2': float(r2), 'RMSE': float(rmse),
            'params': popt,
        }
    except Exception as e:
        results['Langmuir'] = {'error': str(e)}

    # ── Freundlich ────────────────────────────────────────────────────────────
    try:
        log_Ce = np.log(Ce)
        log_q  = np.log(q)
        slope, intercept = np.polyfit(log_Ce, log_q, 1)
        n0  = 1.0 / slope if abs(slope) > 1e-6 else 2.0
        Kf0 = np.exp(intercept)
        p0  = [max(Kf0, 1e-4), max(n0, 0.1)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                freundlich_model, Ce, q,
                p0=p0,
                bounds=([0.0, 0.1], [np.inf, 20.0]),
                maxfev=20000,
            )
        q_pred = freundlich_model(Ce, *popt)
        r2   = r2_score_manual(q, q_pred)
        rmse = np.sqrt(np.mean((q - q_pred) ** 2))
        results['Freundlich'] = {
            'Kf': float(popt[0]), 'n': float(popt[1]),
            'R2': float(r2), 'RMSE': float(rmse),
            'params': popt,
        }
    except Exception as e:
        results['Freundlich'] = {'error': str(e)}

    return results


def plot_isotherms(Ce_array, q_array, results,
                   title="Extraction Isotherm",
                   xlabel="Ce (N) — Equilibrium Aqueous Conc.",
                   ylabel="q (N) — Loading / Corg_eq",
                   save_path=None):
    """Plot Langmuir and Freundlich fits alongside data points."""
    Ce = np.asarray(Ce_array, dtype=float)
    q  = np.asarray(q_array,  dtype=float)
    Ce_smooth = np.linspace(max(Ce.min() * 0.5, 1e-4), Ce.max() * 1.1, 300)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for ax, model_name in zip(axes, ['Langmuir', 'Freundlich']):
        ax.scatter(Ce, q, color='black', s=55, zorder=5, label='Experimental data')
        res = results.get(model_name, {})
        if 'error' not in res:
            if model_name == 'Langmuir':
                q_smooth = langmuir_model(Ce_smooth, *res['params'])
                lbl = (f"Langmuir fit\n"
                       f"$Q_{{max}}$ = {res['Qmax']:.4f}\n"
                       f"$b$ = {res['b']:.4f}\n"
                       f"$R^2$ = {res['R2']:.4f}")
            else:
                q_smooth = freundlich_model(Ce_smooth, *res['params'])
                lbl = (f"Freundlich fit\n"
                       f"$K_f$ = {res['Kf']:.4f}\n"
                       f"$n$ = {res['n']:.4f}\n"
                       f"$R^2$ = {res['R2']:.4f}")
            ax.plot(Ce_smooth, q_smooth, 'r-', linewidth=2, label=lbl)
        else:
            ax.text(0.3, 0.5, f"Fit failed:\n{res['error']}",
                    transform=ax.transAxes, color='red')

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{model_name} Isotherm", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
