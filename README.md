# DS511: Predictive Maintenance of Turbofan Engines
### Applied Multivariate Analysis — Course Project Part 1
**IIT Ropar | MSDSM-03 | AY 2025–2026**

> Swastayan Borah `2025DSS1028`

---

## Overview

End-to-end multivariate regression pipeline for predicting **Remaining Useful Life (RUL)** of turbofan engines using the NASA C-MAPSS benchmark dataset (FD001–FD004). Covers outlier removal, three feature-selection strategies, five regression models, and full diagnostic reporting.

---

## Repository Structure

```
.
├── DS511_Project_Part1.ipynb   # Main notebook (all code)
├── data/
│   ├── train_FD001.txt         # Training trajectories
│   ├── train_FD002.txt
│   ├── train_FD003.txt
│   ├── train_FD004.txt
│   ├── test_FD001.txt          # Test trajectories
│   ├── test_FD002.txt
│   ├── test_FD003.txt
│   ├── test_FD004.txt
│   ├── RUL_FD001.txt           # Ground-truth RUL vectors
│   ├── RUL_FD002.txt
│   ├── RUL_FD003.txt
│   └── RUL_FD004.txt
├── docs/
│   ├── Course_Project_Part_1_DS_5112.pdf
│   ├── SystemProposal_DS511_Final.pdf
│   └── Damage_Propagation_Modeling.pdf
├── README.md
└── .gitignore
```

> **Note:** If you clone this repo without the data files, download the C-MAPSS dataset from the [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) and place the `.txt` files under `data/`. Update the `base=` path in the notebook accordingly.

---

## Dataset: NASA C-MAPSS

| Subset | Train engines | Test engines | Conditions | Fault modes |
|--------|:---:|:---:|:---:|---|
| FD001 | 100 | 100 | 1 (Sea Level) | HPC Degradation |
| FD002 | 260 | 259 | 6 | HPC Degradation |
| FD003 | 100 | 100 | 1 (Sea Level) | HPC + Fan Degradation |
| FD004 | 248 | 249 | 6 | HPC + Fan Degradation |

Each row: 1 engine cycle snapshot → 26 columns (unit, cycle, 3 op. settings, 21 sensors + 1 ignored).  
Target: **RUL** (remaining operational cycles before failure).

---

## Methodology

### Preprocessing
- Drop 6 near-zero-variance sensors (`s1, s5, s10, s16, s18, s19`)
- **Outlier removal:** Median ± 3 × MAD per sensor column
- **Split:** 70% train / 30% validation (stratified by random seed 42)
- **Normalisation:** StandardScaler (fit on train only)

### Feature Selection
| Method | Implementation |
|--------|---------------|
| Forward Selection | `SequentialFeatureSelector` — 5-fold CV, R² scoring |
| Lasso (L1) | `LassoCV` — 5-fold CV; non-zero coefficients retained |
| PCA | Retain PCs explaining ≥ 95% cumulative variance |

### Models
| Model | Features used |
|-------|--------------|
| OLS 1st order | Forward Selection |
| OLS 2nd order | Forward Selection + polynomial expansion |
| Lasso Regression | Lasso-selected features, CV-tuned α |
| Ridge Regression | All features, CV-tuned α |
| PCR | PCA components (95% variance) |

### Evaluation Metrics
- **SSE** — Sum of Squared Errors
- **R²** — Coefficient of determination
- **AIC** — Akaike Information Criterion: `n·ln(SSE/n) + 2k`
- **Residual plots** — residuals vs. sample index
- **ACF of residuals** — autocorrelation up to lag 30 with ±1.96/√n bounds
- **Scatter plots** — ŷ vs. y_test per model

---

## Setup & Usage

### Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn notebook
```

Python ≥ 3.9 recommended.

### Run

```bash
jupyter notebook DS511_Project_Part1.ipynb
```

Execute cells sequentially. Data is loaded from `data/` by default (update `base=` in Cell 2 if needed).

---

## Key Results (summary)

Across all four subsets, **Ridge Regression** and **PCR** consistently yield the lowest AIC and highest R², with residuals closest to white noise. OLS 2nd order achieves the best SSE on single-condition subsets (FD001/FD003) but overfits on multi-condition ones (FD002/FD004). Autocorrelation analysis confirms that all static regression models retain some lag-1 structure — inherent to the time-series nature of degradation — but regularised models (Ridge, PCR) minimise it most effectively within the project scope.

---

## References

1. A. Saxena, K. Goebel, D. Simon, N. Eklund — *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, PHM'08, Denver CO, 2008.
2. NASA Ames Prognostics Center of Excellence — [C-MAPSS Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
