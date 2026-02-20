"""
src/feature_engineering.py
Feature engineering: polynomial features, data splitting, and scaling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    BASE_FEATURES, GAMMA_FEATURES, POLY_FEATURES, FULL_FEATURES,
    TEST_SIZE, RANDOM_SEED
)

FEATURE_SET_MAP = {
    'base':       BASE_FEATURES,
    'with_gamma': GAMMA_FEATURES,
    'poly':       POLY_FEATURES,
    'full':       FULL_FEATURES,
}


def add_polynomial_features(df):
    """
    Add 7 derived polynomial/interaction features to DataFrame.

    Features added:
      Cin_sq, TBA_sq, DES_sq      — squared terms
      Cin_x_TBA, Cin_x_DES,
      TBA_x_DES                   — pairwise interactions
      Cin_x_TBA_x_DES             — three-way interaction
    """
    df = df.copy()
    df['Cin_sq']        = df['Cin'] ** 2
    df['TBA_sq']        = df['TBA_pct'] ** 2
    df['DES_sq']        = df['DES_ratio_num'] ** 2
    df['Cin_x_TBA']     = df['Cin'] * df['TBA_pct']
    df['Cin_x_DES']     = df['Cin'] * df['DES_ratio_num']
    df['TBA_x_DES']     = df['TBA_pct'] * df['DES_ratio_num']
    df['Cin_x_TBA_x_DES'] = df['Cin'] * df['TBA_pct'] * df['DES_ratio_num']
    return df


def get_features(df, feature_set='full'):
    """Return the list of feature column names that exist in df."""
    cols = FEATURE_SET_MAP.get(feature_set, FULL_FEATURES)
    return [c for c in cols if c in df.columns]


def prepare_data(df, target_col, feature_set='full',
                 test_size=TEST_SIZE, random_seed=RANDOM_SEED,
                 scale=True):
    """
    Split and (optionally) scale data.

    Parameters
    ----------
    df : pd.DataFrame — must contain polynomial features already
    target_col : str — target column name
    feature_set : str — one of 'base', 'with_gamma', 'poly', 'full'
    scale : bool — if True, apply StandardScaler to X and y

    Returns
    -------
    X_train_s, X_test_s : scaled arrays
    y_train_s, y_test_s : scaled 1-D arrays
    X_train, X_test     : unscaled arrays (for tree models)
    y_train, y_test     : unscaled 1-D arrays
    scaler_X, scaler_y  : fitted StandardScaler objects
    features            : list of feature column names used
    """
    features = get_features(df, feature_set)
    X = df[features].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    if scale:
        X_train_s = scaler_X.fit_transform(X_train)
        X_test_s  = scaler_X.transform(X_test)
        y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_s  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    else:
        X_train_s = X_train
        X_test_s  = X_test
        y_train_s = y_train
        y_test_s  = y_test
        scaler_X  = None
        scaler_y  = None

    return (X_train_s, X_test_s, y_train_s, y_test_s,
            X_train, X_test, y_train, y_test,
            scaler_X, scaler_y, features)


def build_single_input(Cin, TBA_pct, DES_ratio_num,
                       gamma_aq=None, gamma_org=None, C_TBA_molar=None,
                       feature_set='full'):
    """
    Build a 2-D numpy array (1 row) for a single prediction input.
    Used by the Streamlit app and run_pipeline.py.

    If gamma values are not provided, they are estimated via NRTL model.
    """
    from src.data_generator import compute_nrtl_gamma
    if gamma_aq is None:
        gamma_aq, gamma_org, C_TBA_molar = compute_nrtl_gamma(
            Cin, TBA_pct, DES_ratio_num
        )
    row = {
        'Cin': Cin, 'TBA_pct': TBA_pct, 'DES_ratio_num': DES_ratio_num,
        'gamma_aq': gamma_aq, 'gamma_org': gamma_org, 'C_TBA_molar': C_TBA_molar,
        'Cin_sq': Cin**2, 'TBA_sq': TBA_pct**2, 'DES_sq': DES_ratio_num**2,
        'Cin_x_TBA': Cin*TBA_pct, 'Cin_x_DES': Cin*DES_ratio_num,
        'TBA_x_DES': TBA_pct*DES_ratio_num,
        'Cin_x_TBA_x_DES': Cin*TBA_pct*DES_ratio_num,
    }
    features = get_features(
        pd.DataFrame([row]), feature_set
    )
    return np.array([[row[f] for f in features]]), features
