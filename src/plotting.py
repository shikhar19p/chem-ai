"""
src/plotting.py
Reusable visualisation functions for the reactive extraction ML pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os


def set_global_style():
    """Apply consistent plot style for all figures."""
    plt.rcParams.update({
        'figure.dpi': 120,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
    })


def _save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')


def parity_plot(y_true, y_pred, model_name, target,
                y_std=None, save_path=None):
    """
    Actual vs. Predicted parity plot.
    y_std: optional uncertainty array (for GPR error bars).
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = min(y_true.min(), y_pred.min()) * 0.95
    hi = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Perfect fit')

    if y_std is not None:
        ax.errorbar(y_true, y_pred, yerr=2 * y_std,
                    fmt='o', color='steelblue', alpha=0.7,
                    capsize=3, label='Prediction ± 2σ')
    else:
        ax.scatter(y_true, y_pred, s=55, color='steelblue',
                   edgecolors='navy', alpha=0.8)

    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.set_xlabel(f'Actual {target}')
    ax.set_ylabel(f'Predicted {target}')
    ax.set_title(f'{model_name} — Parity Plot ({target})\n$R^2$ = {r2:.4f}')
    ax.legend()
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def residual_plot(y_true, y_pred, model_name, target, save_path=None):
    """Residuals vs Fitted + histogram of residuals."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].scatter(y_pred, residuals, s=50, color='coral',
                    edgecolors='red', alpha=0.8)
    axes[0].axhline(0, color='black', lw=1.5, linestyle='--')
    axes[0].set_xlabel(f'Fitted {target}')
    axes[0].set_ylabel('Residual (Actual − Predicted)')
    axes[0].set_title(f'{model_name} — Residuals vs Fitted')

    axes[1].hist(residuals, bins=15, color='coral',
                 edgecolor='red', alpha=0.8)
    axes[1].axvline(0, color='black', lw=1.5, linestyle='--')
    axes[1].set_xlabel('Residual')
    axes[1].set_title(f'{model_name} — Residual Distribution')

    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def correlation_heatmap(df, columns=None, title='Feature Correlation', save_path=None):
    """Seaborn heatmap of Pearson correlation coefficients."""
    if columns:
        df = df[columns]
    fig, ax = plt.subplots(figsize=(max(6, len(df.columns) * 0.65),
                                    max(5, len(df.columns) * 0.55)))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                ax=ax, linewidths=0.5)
    ax.set_title(title)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def comparison_heatmap(metrics_df, metric='R2', title=None, save_path=None):
    """Heatmap: Models (rows) × Targets (columns) for one metric."""
    pivot = metrics_df.pivot(index='Model', columns='Target', values=metric)
    fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns) * 1.2),
                                    max(4, len(pivot) * 0.8)))
    cmap = 'RdYlGn' if metric == 'R2' else 'RdYlGn_r'
    sns.heatmap(pivot.astype(float), annot=True, fmt='.4f',
                cmap=cmap, vmin=0 if metric == 'R2' else None,
                vmax=1 if metric == 'R2' else None,
                ax=ax, linewidths=0.5)
    ax.set_title(title or f'{metric} — Model × Target Comparison')
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return pivot


def bar_comparison(metrics_df, metric='RMSE', title=None, save_path=None):
    """Grouped bar chart of one metric across models and targets."""
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = metrics_df.pivot(index='Target', columns='Model', values=metric)
    pivot.plot(kind='bar', ax=ax, colormap='tab10', edgecolor='black', width=0.7)
    ax.set_xlabel('Target')
    ax.set_ylabel(metric)
    ax.set_title(title or f'{metric} Comparison Across Models')
    ax.legend(title='Model', bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def learning_curve_ann(history, target, save_path=None):
    """Plot ANN training + validation loss and MAE curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['loss'],     label='Train MSE', color='steelblue')
    axes[0].plot(history.history['val_loss'], label='Val MSE',   color='orange')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
    axes[0].set_title(f'ANN Learning Curves — {target}')
    axes[0].legend(); axes[0].set_yscale('log')

    if 'mae' in history.history:
        axes[1].plot(history.history['mae'],     label='Train MAE', color='steelblue')
        axes[1].plot(history.history['val_mae'], label='Val MAE',   color='orange')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
        axes[1].set_title(f'ANN MAE Curves — {target}')
        axes[1].legend()

    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
