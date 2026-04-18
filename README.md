# 🛩️ Predictive Maintenance of Turbofan Engines
### Remaining Useful Life (RUL) Prediction via Multivariate Regression
#### DS511: Applied Multivariate Analysis — Course Project Part 1
**Indian Institute of Technology Ropar | MSDSM-03 | AY 2025–2026**

> **Swastayan Borah** `2025DSS1028` &nbsp;·&nbsp; **Vatsal Goswami** `2025DSS1031`

---

## 📌 Table of Contents

1. [What is this project about?](#-what-is-this-project-about)
2. [Why does it matter?](#-why-does-it-matter)
3. [Dataset](#-dataset-nasa-c-mapss)
4. [Project Pipeline](#-project-pipeline)
5. [Methodology Deep Dive](#-methodology-deep-dive)
6. [Models Built](#-models-built)
7. [Evaluation Metrics](#-evaluation-metrics)
8. [Repository Structure](#-repository-structure)
9. [Setup & Installation](#-setup--installation)
10. [How to Run](#-how-to-run)
11. [Results Summary](#-results-summary)
12. [Key Inferences](#-key-inferences)
13. [References](#-references)

---

## 🔍 What is this project about?

This project builds a **data-driven predictive maintenance system** for commercial aircraft turbofan engines. Specifically, we predict the **Remaining Useful Life (RUL)** — the number of operational cycles an engine can still safely run before it fails.

We treat this as a **multivariate regression problem**: given a snapshot of 24 sensor/setting measurements from an engine at a given cycle, predict how many cycles remain before failure.

The entire pipeline — from raw sensor data to a validated regression model — is implemented from scratch using classical statistical learning methods as specified in the DS511 course project guidelines.

---

## 💡 Why does it matter?

Aircraft engines are among the most safety-critical and expensive mechanical systems in the world.

- ✈️ An **unplanned engine failure** can cost airlines between **$10,000–$150,000 per hour** of unscheduled downtime
- 🔧 **Over-maintenance** (replacing parts too early) wastes millions annually
- ⚠️ **Under-maintenance** (missing degradation) risks catastrophic failure

Accurate RUL prediction enables **condition-based maintenance** — servicing an engine exactly when needed, not too early and not too late. This is the core promise of **Prognostics and Health Management (PHM)**.

---

## 📦 Dataset: NASA C-MAPSS

The dataset used is the **Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)**, publicly available from the [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

### What is C-MAPSS?

C-MAPSS is a high-fidelity simulation of a **90,000 lb thrust class turbofan engine** modelled in MATLAB/Simulink. It simulates realistic engine degradation by injecting faults into engine subsystems (like the High-Pressure Compressor) and running the engine until failure. Each "run" produces a multivariate time series of sensor readings — one row per operational cycle.

### Engine Architecture

The engine has five rotating components: **Fan → LPC → HPC → HPT → LPT** (Low/High Pressure Compressor/Turbine). Faults are introduced into one or two of these modules depending on the subset.

### The Four Subsets

| Subset | Training Engines | Test Engines | Operating Conditions | Fault Modes |
|--------|:---:|:---:|:---:|---|
| **FD001** | 100 | 100 | 1 (Sea Level) | HPC Degradation only |
| **FD002** | 260 | 259 | 6 (varying altitude, Mach, throttle) | HPC Degradation only |
| **FD003** | 100 | 100 | 1 (Sea Level) | HPC + Fan Degradation |
| **FD004** | 248 | 249 | 6 (varying) | HPC + Fan Degradation |

- FD001/FD003 are **simpler** (single operating condition)
- FD002/FD004 are **harder** (6 flight envelopes, engines switch between conditions each cycle)

### Data Format

Each file is a space-separated text file with **26 columns**:

| Column(s) | Description |
|-----------|-------------|
| 1 | Unit (engine) number |
| 2 | Time in cycles |
| 3–5 | Operational settings: altitude, Mach number, throttle resolver angle (TRA) |
| 6–26 | Sensor measurements (temperatures, pressures, speeds, fuel flow, etc.) |

The **training set** runs each engine until failure. The **test set** stops each engine some cycles before failure. The `RUL_FD00X.txt` files provide the true remaining cycles at the last observed cycle for each test engine.

### RUL Labelling

- **Training:** `RUL = max_cycle_for_that_engine − current_cycle`
- **Test:** `RUL = (max_cycle_in_test_for_that_engine − current_cycle) + ground_truth_RUL`

---

## 🔄 Project Pipeline

```
Raw Data (train/test .txt)
        │
        ▼
  ┌─────────────────────────┐
  │   DATA PREPROCESSING    │
  │  • Drop constant sensors│
  │  • Outlier removal (MAD)│
  │  • 70/30 split          │
  │  • StandardScaler       │
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │   FEATURE SELECTION     │
  │  • Forward Selection    │
  │  • Lasso (L1)           │
  │  • PCA (95% variance)   │
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │    MODEL BUILDING       │
  │  • OLS 1st order        │
  │  • OLS 2nd order        │
  │  • Lasso Regression     │
  │  • Ridge Regression     │
  │  • PCR                  │
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │      VALIDATION         │
  │  • SSE, R², AIC         │
  │  • Residual plots       │
  │  • ACF of residuals     │
  │  • ŷ vs y scatter       │
  └─────────────────────────┘
```

---

## 🔬 Methodology Deep Dive

### Step 1 — Sensor Filtering

Six sensors (`s1, s5, s10, s16, s18, s19`) are near-constant across all engines and cycles — they carry no degradation signal and only add noise to the model. These are dropped upfront, leaving **18 input features** (3 operational settings + 15 sensor channels).

### Step 2 — Outlier Removal (Median ± 3×MAD)

Standard z-score outlier removal assumes a normal distribution, which is not guaranteed for engine sensor data. We use the more robust **Median Absolute Deviation (MAD)**:

```
MAD = median( |xᵢ − median(X)| )
Keep row if: median(X) − 3·MAD ≤ xᵢ ≤ median(X) + 3·MAD
```

This is applied **per sensor column** across the entire training set, removing rows where any sensor reading is an extreme anomaly.

### Step 3 — Train/Validation Split (70/30)

The cleaned training data is split 70% for model fitting and 30% for validation. The split is done on **rows** (individual cycle snapshots), not on whole engines — consistent with the project specification.

### Step 4 — Normalisation

Each feature is scaled to **zero mean and unit variance** using `StandardScaler`. The scaler is fitted **only on the training split** and then applied to the validation set — preventing data leakage.

### Step 5 — Feature Selection (Three Methods)

#### A. Forward Selection
Starts with an empty feature set and greedily adds one feature at a time — the one that most improves 5-fold cross-validated R². Stops when no further improvement is possible. This is a wrapper method: it evaluates features by how well a linear model uses them.

#### B. Lasso (L1 Penalised Regression)
Lasso adds an L1 penalty (`α·Σ|βⱼ|`) to the OLS loss function. This penalty drives irrelevant coefficients to **exactly zero**, performing automatic feature selection. The regularisation parameter α is chosen via 5-fold cross-validation (`LassoCV`). Only features with non-zero coefficients are retained.

#### C. PCA-Based Selection
Principal Component Analysis finds orthogonal directions of maximum variance in the feature space. We retain enough principal components to explain **≥95% of total variance** in the training data. This removes multicollinearity entirely (all PCs are uncorrelated) at the cost of interpretability.

---

## 🤖 Models Built

### 1. OLS 1st Order (Linear Regression)
Standard ordinary least squares on the features selected by Forward Selection. Minimises `Σ(yᵢ − ŷᵢ)²` with no regularisation. Provides a baseline.

### 2. OLS 2nd Order (Polynomial Regression)
Expands the Forward-selected features into all degree-2 polynomial terms (squares + cross-products), then fits OLS. Captures nonlinear degradation trends. Risks overfitting on multi-condition subsets.

### 3. Lasso Regression
Fits a regularised regression on Lasso-selected features with the CV-optimal α. The L1 penalty simultaneously selects features and shrinks coefficients, improving generalisation on noisy sensor data.

### 4. Ridge Regression (Bayesian Linear Regression)
Adds an L2 penalty (`α·Σβⱼ²`) to OLS. Unlike Lasso it does **not** zero out coefficients — it shrinks all of them proportionally. This is the Bayesian equivalent of placing a Gaussian prior on the weights. Particularly effective when many correlated sensors all carry partial information. α chosen via `RidgeCV`.

### 5. Principal Component Regression (PCR)
Transforms the feature space using PCA (from Step 5C) and then fits OLS on the principal components. Because PCs are orthogonal, there is zero multicollinearity — a major source of instability in OLS on raw sensor data.

---

## 📊 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **SSE** | `Σ(yᵢ − ŷᵢ)²` | Total squared prediction error — lower is better |
| **R²** | `1 − SSE/SST` | Proportion of variance explained — higher is better (max 1.0) |
| **AIC** | `n·ln(SSE/n) + 2k` | Penalises model complexity — lower is better |

Where `k` = number of model parameters (coefficients + intercept) and `n` = number of validation samples.

### Diagnostic Plots (per model, per FD subset)

1. **Residuals vs. sample index** — should scatter randomly around zero; any pattern indicates a missed structure
2. **Autocorrelation of residuals (ACF)** — bars exceeding ±1.96/√n bounds indicate the model has not captured temporal dependencies
3. **ŷ vs. y_test scatter** — points should lie along the 45° diagonal; systematic deviations reveal bias

---

## 📁 Repository Structure

```
ds511-cmapss-rul/
│
├── DS511_Project_Part1.ipynb   ← Main notebook: all code, plots, analysis
│
├── CMAPSSData/                 ← Raw data (download from NASA, not tracked by git)
│   ├── train_FD001.txt
│   ├── train_FD002.txt
│   ├── train_FD003.txt
│   ├── train_FD004.txt
│   ├── test_FD001.txt
│   ├── test_FD002.txt
│   ├── test_FD003.txt
│   ├── test_FD004.txt
│   ├── RUL_FD001.txt
│   ├── RUL_FD002.txt
│   ├── RUL_FD003.txt
│   └── RUL_FD004.txt
│
├── docs/                       ← Project documentation
│   ├── Course_Project_Part_1_DS_5112.pdf
│   ├── SystemProposal_DS511_Final.pdf
│   └── Damage_Propagation_Modeling.pdf
│
├── README.md                   ← This file
└── .gitignore
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9 or higher
- `pip` package manager
- Jupyter Notebook or JupyterLab

### Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn notebook
```

Or with a virtual environment (recommended):

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install numpy pandas scikit-learn matplotlib seaborn notebook
```

### Get the data

Download the C-MAPSS dataset from the NASA Prognostics repository:
👉 https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

Place all `.txt` files inside a folder named `CMAPSSData/` at the project root.

---

## ▶️ How to Run

```bash
git clone https://github.com/<your-username>/ds511-cmapss-rul.git
cd ds511-cmapss-rul
jupyter notebook DS511_Project_Part1.ipynb
```

Run cells **sequentially from top to bottom**. Each section is self-contained with markdown explanations. The notebook automatically processes all four FD subsets (FD001–FD004) in every step.

> If your data is in a different folder, update the `base=` argument in the `load_dataset()` call in Cell 2.

---

## 📈 Results Summary

Across all four subsets, **Ridge Regression** and **PCR** consistently deliver the best balance of low AIC, high R², and low residual autocorrelation.

| Model | Strength | Weakness |
|-------|----------|----------|
| OLS 1st order | Fast, interpretable baseline | Misses nonlinear degradation trends |
| OLS 2nd order | Captures curvature, best SSE on FD001/FD003 | Overfits on multi-condition FD002/FD004 |
| Lasso | Sparse, robust to irrelevant sensors | Underfits slightly due to aggressive shrinkage |
| **Ridge** | **Best overall generalisation** | Coefficients less interpretable than OLS |
| **PCR** | **Eliminates multicollinearity entirely** | PCs are not physically interpretable |

FD001 and FD003 (single operating condition) are significantly easier — all models perform well. FD002 and FD004 (6 operating conditions) are substantially harder; Ridge and PCR maintain performance while OLS variants degrade.

---

## 🧠 Key Inferences

**From residual plots:**
All models show some heteroscedasticity — residuals tend to be larger at high RUL values (early in engine life) and smaller near end-of-life. This is expected: predicting far into the future is inherently harder than predicting imminent failure.

**From ACF plots:**
Lag-1 autocorrelation persists across all models, especially on FD002/FD004. This means the models are not fully capturing the temporal degradation dynamics — a fundamental limitation of static (non-temporal) regression. Time-series methods (e.g. LSTM, state-space models) would address this but are beyond the scope of Part 1.

**From ŷ vs y scatter:**
Ridge and PCR scatter plots show the tightest clustering around the 45° line. OLS 2nd order occasionally produces negative RUL predictions on multi-condition subsets — a sign of overfitting.

---

## 📚 References

1. A. Saxena, K. Goebel, D. Simon, N. Eklund — *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, 1st International Conference on Prognostics and Health Management (PHM'08), Denver CO, Oct 2008.
2. NASA Ames Prognostics Center of Excellence — C-MAPSS Dataset. https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
3. Hastie, T., Tibshirani, R., Friedman, J. — *The Elements of Statistical Learning*, 2nd ed., Springer, 2009.

---

<p align="center">
  <sub>IIT Ropar · DS511 Applied Multivariate Analysis · 2025–2026</sub>
</p>
