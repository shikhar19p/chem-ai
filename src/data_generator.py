"""
src/data_generator.py
Synthetic dataset generation using simplified NRTL activity coefficient model.
Based on: Wasewar et al. (2002), Uslu & Kirbaslar (2008) for similar
carboxylic acid + amine + organic solvent reactive extraction systems.
"""

import numpy as np
import pandas as pd
import os
import sys

# Allow running standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    CIN_LEVELS, TBA_LEVELS, DES_RATIO_LEVELS, TEMPERATURE_K,
    NRTL_TAU_AW, NRTL_TAU_WA, NRTL_ALPHA, NRTL_K_EX_BASE,
    TBA_MW, DES_BASE_DENSITY, RANDOM_SEED, SYN_DATA_CSV, RAW_DATA_CSV
)


def compute_nrtl_gamma(Cin, TBA_pct, DES_ratio_num, T=TEMPERATURE_K):
    """
    Estimate activity coefficients using a simplified NRTL-style model.

    Parameters
    ----------
    Cin : float  - initial acid concentration (N)
    TBA_pct : float  - TBA wt% in organic phase
    DES_ratio_num : float  - DES molar ratio encoding (1.0, 1.5, or 2.0)
    T : float  - temperature (K)

    Returns
    -------
    gamma_aq : float  - activity coefficient of acid in aqueous phase
    gamma_org : float  - activity coefficient of acid in organic phase
    C_TBA_molar : float  - molar concentration of TBA in organic phase (mol/L)
    """
    # Mole fraction of acid in aqueous phase (dilute, mostly water)
    x_acid_aq = Cin / (Cin + 55.5)   # 55.5 mol/L = molarity of water

    # Temperature-corrected NRTL interaction parameters
    dT = T - 298.0
    tau_aw = NRTL_TAU_AW - 0.003 * dT
    tau_wa = NRTL_TAU_WA - 0.001 * dT

    G_aw = np.exp(-NRTL_ALPHA * tau_aw)
    G_wa = np.exp(-NRTL_ALPHA * tau_wa)
    x_w = 1.0 - x_acid_aq

    # ln(gamma) for acid in aqueous phase (NRTL combinatorial + residual)
    denom1 = x_acid_aq + x_w * G_wa
    denom2 = x_w + x_acid_aq * G_aw
    ln_gamma_aq = 0.0
    if denom1 > 1e-12:
        ln_gamma_aq += x_w ** 2 * tau_wa * (G_wa / denom1) ** 2
    if denom2 > 1e-12:
        ln_gamma_aq += x_w ** 2 * tau_aw * G_aw / denom2 ** 2
    gamma_aq = np.exp(np.clip(ln_gamma_aq, -3.0, 3.0))

    # Organic phase: TBA volume fraction effect + DES polarity modulation
    # Higher DES_ratio_num (thymol-rich) → less polar → better solvation of complex
    phi_TBA = TBA_pct / 100.0
    DES_polarity_factor = 1.0 - 0.10 * (DES_ratio_num - 1.0)
    tau_org = 0.8 * DES_polarity_factor - 0.002 * dT
    ln_gamma_org = (1.0 - phi_TBA) ** 2 * tau_org
    gamma_org = np.exp(np.clip(ln_gamma_org, -2.0, 2.0))

    # TBA molar concentration in organic phase
    rho_org = DES_BASE_DENSITY + 0.01 * DES_ratio_num  # approximate density g/mL
    C_TBA_molar = (TBA_pct / 100.0) * rho_org * 1000.0 / TBA_MW  # mol/L

    return gamma_aq, gamma_org, C_TBA_molar


def compute_kd(Cin, TBA_pct, DES_ratio_num, gamma_aq, gamma_org, C_TBA_molar,
               noise_pct=0.04, rng=None):
    """
    Distribution coefficient from modified chemical equilibrium model.

    KD = K_ex * C_TBA_molar * (gamma_aq / gamma_org)

    K_ex for TBA + propionic acid ≈ 20–40 L/mol (Wasewar et al.),
    modulated by DES molar ratio (thymol-rich DES enhances ion-pair solvation).
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    K_ex = NRTL_K_EX_BASE * (1.0 + 0.15 * (DES_ratio_num - 1.0))
    KD = K_ex * C_TBA_molar * (gamma_aq / gamma_org)
    noise = rng.normal(0.0, noise_pct * abs(KD))
    return max(0.01, KD + noise)


def compute_all_targets(Cin, TBA_pct, DES_ratio_num, KD, C_TBA_molar):
    """
    Compute E_pct, Corg_eq, Caq_eq, Z, SF_min from KD.
    Assumes equal O/A volume ratio (phi = 1).
    """
    phi = 1.0  # O:A volume ratio
    E_pct = 100.0 * KD / (KD + phi)
    Corg_eq = Cin * (E_pct / 100.0)
    Caq_eq = max(Cin - Corg_eq, 0.0)
    Z = Corg_eq / max(C_TBA_molar, 1e-6)
    SF_min = (1.0 / max(KD, 0.01)) * (1.0 + 0.1 * Cin)
    return {
        'E_pct': float(np.clip(E_pct, 0.0, 100.0)),
        'Corg_eq': float(Corg_eq),
        'Caq_eq': float(Caq_eq),
        'Z': float(max(Z, 0.0)),
        'SF_min': float(SF_min),
    }


def generate_synthetic_dataset(n_noise_runs=3, random_seed=RANDOM_SEED):
    """
    Generate a full-factorial synthetic dataset from literature-based model.

    Full factorial: 4 Cin × 4 TBA × 3 DES_ratio = 48 base rows.
    With n_noise_runs=3, produces 144 rows (realistic replication).

    Returns
    -------
    pd.DataFrame with columns:
        Cin, TBA_pct, DES_ratio_num, Temperature_K,
        gamma_aq, gamma_org, C_TBA_molar,
        KD, E_pct, Corg_eq, Caq_eq, Z, SF_min
    """
    rng = np.random.default_rng(random_seed)
    rows = []
    T = TEMPERATURE_K

    for run in range(n_noise_runs):
        for Cin in CIN_LEVELS:
            for TBA_pct in TBA_LEVELS:
                for DES_ratio_num in DES_RATIO_LEVELS:
                    ga, go, C_TBA = compute_nrtl_gamma(Cin, TBA_pct, DES_ratio_num, T)
                    KD = compute_kd(Cin, TBA_pct, DES_ratio_num, ga, go, C_TBA,
                                    noise_pct=0.04, rng=rng)
                    tgts = compute_all_targets(Cin, TBA_pct, DES_ratio_num, KD, C_TBA)
                    rows.append({
                        'Cin': Cin,
                        'TBA_pct': float(TBA_pct),
                        'DES_ratio_num': DES_ratio_num,
                        'Temperature_K': T,
                        'gamma_aq': ga,
                        'gamma_org': go,
                        'C_TBA_molar': C_TBA,
                        'KD': KD,
                        **tgts,
                    })

    return pd.DataFrame(rows)


def predict_single(Cin, TBA_pct, DES_ratio_num, T=TEMPERATURE_K):
    """
    Quick NRTL-model prediction for a single condition (no noise).
    Used by the Streamlit app as a sanity reference.
    """
    ga, go, C_TBA = compute_nrtl_gamma(Cin, TBA_pct, DES_ratio_num, T)
    KD = compute_kd(Cin, TBA_pct, DES_ratio_num, ga, go, C_TBA, noise_pct=0.0)
    tgts = compute_all_targets(Cin, TBA_pct, DES_ratio_num, KD, C_TBA)
    return {'KD': KD, **tgts, 'gamma_aq': ga, 'gamma_org': go, 'C_TBA_molar': C_TBA}


def load_or_generate(csv_path=RAW_DATA_CSV, random_seed=RANDOM_SEED):
    """
    Try to load real thesis data CSV; fall back to synthetic generation.

    Returns
    -------
    df : pd.DataFrame
    source : str  - 'real' or 'synthetic'
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Fill thermodynamic columns if not present
        if 'gamma_aq' not in df.columns:
            gammas = df.apply(
                lambda r: compute_nrtl_gamma(
                    r['Cin'], r['TBA_pct'], r['DES_ratio_num']
                ), axis=1
            )
            df['gamma_aq']    = gammas.apply(lambda x: x[0])
            df['gamma_org']   = gammas.apply(lambda x: x[1])
            df['C_TBA_molar'] = gammas.apply(lambda x: x[2])
        print(f"[data] Loaded real data: {len(df)} rows from {csv_path}")
        return df, 'real'
    else:
        print(f"[data] thesis_data.csv not found. Generating synthetic dataset...")
        df = generate_synthetic_dataset(random_seed=random_seed)
        os.makedirs(os.path.dirname(SYN_DATA_CSV), exist_ok=True)
        df.to_csv(SYN_DATA_CSV, index=False)
        print(f"[data] Synthetic dataset saved: {len(df)} rows -> {SYN_DATA_CSV}")
        return df, 'synthetic'


# ─── Standalone test ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    df, source = load_or_generate()
    print(f"\nSource: {source}")
    print(df.describe().round(4))
    print(f"\nFirst 5 rows:\n{df.head()}")
