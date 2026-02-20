# Reactive Extraction Optimizer

**ML/DL Ensemble Predictor for Reactive Liquid-Liquid Extraction**
System: Propionic Acid · TBA (Tri-n-Butylamine) · DES (Thymol:Menthol)

## 🌐 Live Demo

**Frontend (GitHub Pages):** https://shikhar19p.github.io/chem-ai/

> ⚠️ The live demo shows the UI. For live predictions, run the Python backend locally (see below).

---

## 🚀 Features

- **5 ML/DL Models:** RSM+ANOVA, Random Forest, XGBoost, GPR (Gaussian Process), ANN (Neural Network)
- **4 Prediction Targets:** KD (distribution coefficient), E% (extraction efficiency), Z (loading ratio), SF_min
- **NRTL Thermodynamics:** Activity coefficients computed via NRTL model
- **Sensitivity Analysis:** Sweep any input variable while holding others fixed
- **2D Response Matrix:** Heatmap of any two input variables vs any target
- **Adsorption Isotherms:** Langmuir and Freundlich fitting with R² display
- **Chemistry Database:** 12 acids with molecular structures, properties, stoichiometry, intermediates
- **Bayesian Optimisation:** GPR-based next-experiment suggestions (UCB acquisition)

---

## 📁 Project Structure

```
chem-ai/
├── index.html              # Frontend (GitHub Pages served)
├── frontend/
│   └── index.html          # React SPA (Babel, Chart.js)
├── api.py                  # FastAPI backend (port 8000)
├── config.py               # Constants & hyperparameters
├── src/
│   ├── data_generator.py   # NRTL synthetic data
│   ├── feature_engineering.py
│   ├── isotherm_fitting.py # Langmuir & Freundlich
│   ├── metrics.py
│   └── models/
│       ├── regression.py   # RSM + ANOVA
│       ├── random_forest.py
│       ├── xgboost_model.py
│       ├── gpr_model.py    # GPR + Bayesian Opt
│       └── ann_model.py    # Keras ANN
├── run_pipeline.py         # CLI entry point
└── launch_app.bat          # One-click launcher (Windows)
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install fastapi uvicorn scikit-learn xgboost tensorflow joblib bayesian-optimization statsmodels shap pandas numpy scipy matplotlib seaborn
```

### 2. Train models
```bash
python run_pipeline.py --data synthetic --target all
```

### 3. Start API server
```bash
uvicorn api:app --reload --port 8000
```

### 4. Open frontend
Open `frontend/index.html` in your browser (or double-click `launch_app.bat`).

---

## 🧪 System Details

| Parameter | Range |
|-----------|-------|
| Initial Conc. (Cin) | 0.05 – 0.20 N (slider: 0.01 – 1.0 N) |
| TBA wt% | 5, 10, 15, 20% (slider: 1 – 100%) |
| DES Ratio (Thymol:Menthol) | 1:1 / 1:1.5 / 2:1 (or custom) |
| Temperature | 306 K |
| Pressure | 101.32 kPa |
| O/A Ratio | 1:1 |

---

## 📊 Model Performance (Synthetic Data)

| Model | R² (avg) |
|-------|----------|
| Random Forest | 0.9973 |
| XGBoost | 0.9974 |
| GPR | 0.9970 |
| RSM | 0.9867 |
| ANN | 0.9743 |

---

## 🔬 Chemistry

The system models reactive extraction where:

```
AH(aq) + TBA(org) ⇌ [A⁻·TBAH⁺](org)
```

**KD = K_ex × C_TBA_molar × (γ_aq / γ_org)**

- γ values from NRTL activity coefficient model
- K_ex_base = 28 L/mol (literature, carboxylic acid/amine systems)

---

*Chemical Engineering Thesis Project — ML/DL Component*
