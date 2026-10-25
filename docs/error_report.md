# Comprehensive Error Calculation & Diagnostics Report (Agent 7)

**Project Title:** Machine Learning Benchmark for Soil Fertility, Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal  
**Analytical Scope:** Complete 20-Error Framework, 12 Soil Parameter Predictability Analysis, and Overfit/Underfit Diagnosis  
**Execution Timestamp:** 2026-10-24  
**Primary Deliverables:** `error_master_table.xlsx` | `error_analysis_12_parameters.xlsx` | `error_report.md`

---

## 1. Executive Error Summary

Across 24 supervised classification models and 12 regression models:
1. **Winning Classification Model:** **Extra Trees Classifier ($n=100$)** achieved the lowest test classification error ($0.5705$, test accuracy $42.95\%$) and lowest aggregate penalty under conservative 4:1 agronomic risk ($17,115.0$).
2. **Winning Yield Regressor:** **Lasso Regressor** achieved $R^2 = 0.9086$, $RMSE = 5,057.14\text{ kg/ha}$ (normalized $RMSE = 6.29\%$), and a minimal generalization gap ($24.45\text{ kg/ha}$).
3. **The 12-Parameter Predictability Gradient:**
   - **Predictable Soil Attributes ($R^2 > 0$):** Potassium ($K$, $R^2 = 0.2086$, $nRMSE = 14.13\%$), Available Nitrogen ($N$, $R^2 = 0.1383$, $nRMSE = 14.52\%$), Organic Carbon ($OC$, $R^2 = 0.1009$, $nRMSE = 16.09\%$), and Soil Reaction ($pH$, $R^2 = 0.0982$, $nRMSE = 18.01\%$).
   - **Unpredictable / Stochastic Micronutrients ($R^2 \le 0$):** Zinc ($Zn$, $R^2 = -0.0017$), Phosphorus ($P$, $R^2 = -0.0028$), Iron ($Fe$, $R^2 = -0.0028$), Sulphur ($S$, $R^2 = -0.0033$), Copper ($Cu$, $R^2 = -0.0038$), Manganese ($Mn$, $R^2 = -0.0052$), Boron ($B$, $R^2 = -0.0025$), and Electrical Conductivity ($EC$, $R^2 = -0.0040$).

---

## 2. Part A: The 12 Soil Parameter Error Analysis

A core empirical objective was evaluating whether each of the 12 core soil physicochemical parameters can be predicted from the remaining 11 parameters plus geospatial and meteorological attributes:

| # | Parameter | Symbol | Unit | Ridge $R^2$ | RF $R^2$ | HistGBM $R^2$ | HistGBM RMSE | Normalized RMSE ($nRMSE$) | Mean Bias Error (MBE) | Predictability Level |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Available Potassium | $K$ | kg/ha | 0.1985 | 0.1850 | **0.2086** | 44.86 kg/ha | **14.13%** | +0.12 | **Moderate / High** |
| 2 | Available Nitrogen | $N$ | kg/ha | 0.1250 | 0.1180 | **0.1383** | 56.12 kg/ha | **14.52%** | -0.28 | **Moderate** |
| 3 | Organic Carbon | $OC$ | % | 0.0920 | 0.0880 | **0.1009** | 0.221 % | **16.09%** | +0.001 | **Moderate** |
| 4 | Soil Reaction | $pH$ | scale | 0.0715 | 0.0700 | **0.0982** | 0.811 | **18.01%** | +0.004 | **Moderate** |
| 5 | Available Zinc | $Zn$ | mg/kg | -0.0012 | -0.0120 | -0.0017 | 0.342 mg/kg | 15.22% | +0.001 | Stochastic / Unpredictable |
| 6 | Available Phosphorus| $P$ | kg/ha | -0.0015 | -0.0150 | -0.0028 | 11.83 kg/ha | 15.35% | -0.015 | Chemical Fixation |
| 7 | Available Iron | $Fe$ | mg/kg | -0.0018 | -0.0140 | -0.0028 | 6.86 mg/kg | 15.74% | -0.008 | Stochastic / Unpredictable |
| 8 | Available Sulphur | $S$ | mg/kg | -0.0021 | -0.0180 | -0.0033 | 7.88 mg/kg | 15.76% | +0.012 | Stochastic / Unpredictable |
| 9 | Available Copper | $Cu$ | mg/kg | -0.0025 | -0.0210 | -0.0038 | 0.394 mg/kg | 15.88% | -0.002 | Stochastic / Unpredictable |
| 10 | Available Manganese | $Mn$ | mg/kg | -0.0030 | -0.0250 | -0.0052 | 2.96 mg/kg | 15.51% | +0.003 | Stochastic / Unpredictable |
| 11 | Available Boron | $B$ | mg/kg | -0.0019 | -0.0160 | -0.0025 | 0.248 mg/kg | 16.99% | +0.001 | Stochastic / Unpredictable |
| 12 | Electrical Cond. | $EC$ | dS/m | -0.0004 | -0.0539 | -0.0040 | 0.387 dS/m | 13.19% | +0.001 | Localized / Coastal |

### Geochemical Rationale: Hardest vs Easiest Parameters
- **Why Potassium ($K$), Nitrogen ($N$), and Organic Carbon ($OC$) are Predictable:**
  - Soil Organic Carbon and Nitrogen decompose at rates governed by regional hydro-thermal regimes (mean temperature and rainfall).
  - Potassium availability is anchored in parent rock mineralogy across broad pedological zones (e.g., illite-rich Gangetic alluvium vs weathered kaolinitic laterites of Purulia and Bankura).
- **Why Phosphorus ($P$) and Micronutrients ($Zn, B, Fe, Cu, Mn$) Cannot be Predicted ($R^2 \le 0$):**
  - **Phosphorus Fixation:** In acidic soils ($pH < 5.5$), phosphorus precipitates as insoluble aluminum and iron phosphates; in neutral/alkaline soils, it binds to calcium. Available phosphorus is therefore a non-linear chemical equilibrium state rather than a continuous regional gradient.
  - **Micro-Scale Anthropogenic Management:** Topsoil micronutrients reflect localized management (FYM farmyard manure application, zinc sulphate broadcasting, burning of crop residues) which broad satellite and meteorological variables cannot infer.
  - **Policy Recommendation:** Direct laboratory soil testing is indispensable; machine learning cannot replace wet-chemistry soil testing for micronutrients.

---

## 3. Part B: The 20 Fundamental Error Breakdown for Supervised Models

| # | Error Metric | Classification Application (Crop Suitability) | Regression Application (Yield / Fertilizer) |
|---|---|---|---|
| 1 | **Bias Error** | 0-1 loss bootstrap decomposition ($0.9160 - 0.9210$ for top classifiers) | Squared bias ($2.56 \times 10^7$ for Lasso yield regression) |
| 2 | **Variance Error** | Bootstrap prediction variance ($0.8590 - 0.8650$) | Prediction variance ($2.30 \times 10^5$ for Lasso yield regression) |
| 3 | **Training Error** | $1 - \text{Train Accuracy}$ ($0.0$ for Extra Trees, $0.3755$ for HistGBM) | Train RMSE ($5,032.69\text{ kg/ha}$ for Lasso) |
| 4 | **Validation Error** | 5-Fold CV Error ($0.5861 \pm 0.0056$ for Extra Trees) | 5-Fold CV RMSE ($5,044.72 \pm 41.90\text{ kg/ha}$) |
| 5 | **Test Error** | $1 - \text{Test Accuracy}$ ($0.5705$ for Extra Trees, $42.95\%$ accuracy) | Held-out 30% Test RMSE ($5,057.14\text{ kg/ha}$) |
| 6 | **Generalization Error** | Bootstrap 95% CI: $[0.5433, 0.6000]$; Test-Train gap $= 0.5705$ | Test-Train gap $= 24.45\text{ kg/ha}$ ($0.48\%$ error difference) |
| 7 | **Overfitting** | Overfitting on deep leaf nodes (Extra Trees, CART); Tight for Linear/HistGBM | Well-Regularized (Generalization gap $< 25\text{ kg/ha}$) |
| 8 | **Underfitting** | Severe underfitting for KNN ($18.7\%$) and LDA ($12.8\%$); Optimal for ensembles | Severe underfitting for KNN Regressor ($R^2 = 0.6520$) |
| 9 | **MAE** | `NOT APPLICABLE: Multi-class discrete target` | $3,234.61\text{ kg/ha}$ (Lasso yield regression) |
| 10 | **MSE** | `NOT APPLICABLE: Multi-class discrete target` | $25,574,649.32\text{ kg}^2/\text{ha}^2$ |
| 11 | **RMSE** | `NOT APPLICABLE: Multi-class discrete target` | $5,057.14\text{ kg/ha}$; Normalized RMSE = $6.29\%$ |
| 12 | **MAPE** | `NOT APPLICABLE: Multi-class discrete target` | $132.07\%$ (Driven by near-zero marginal pulse yields) |
| 13 | **Classification Error**| $0.5705$ (Test accuracy $42.95\%$); Balanced Error = $0.5733$ | `NOT APPLICABLE: Continuous regression target` |
| 14 | **False Positive (Type I)**| Total FP = $3,423$; Mean FPR = $0.0519$ (Extra Trees) | `NOT APPLICABLE: Continuous target` |
| 15 | **False Negative (Type II)**| Total FN = $3,423$; Mean FNR = $0.5733$ (Extra Trees) | `NOT APPLICABLE: Continuous target` |
| 16 | **Log Loss** | Raw Log Loss = $1.1702$; Brier Score (Class 0) = $0.0675$ | `NOT APPLICABLE: Minimizes distance, not probability` |
| 17 | **Hinge Loss** | Linear SVM Hinge Loss = $0.8124$; `NOT APPLICABLE` for non-margin models | `NOT APPLICABLE: No margin optimization` |
| 18 | **Residual Error** | `NOT APPLICABLE: Categorical classification` | Mean Residual $= +7.50\text{ kg/ha}$; Durbin-Watson $= 1.973$ |
| 19 | **Sampling Error** | Bootstrap SE = $0.0139$; 5-Fold CV Std = $0.0056$ | CV Fold RMSE Std $= 41.90\text{ kg/ha}$ |
| 20 | **Data Leakage Error**| Clean vs Leaky Preprocessing Inflation = $-0.00017$ | Clean vs Leaky Preprocessing Inflation $= 0.0$ |

---

## 4. Master Diagnostic Takeaways for Publication

1. **The Sub-Tropical Bayes Error Floor:** Multi-crop classification accuracy is constrained to $\sim 43\%$ not by algorithmic deficiencies, but by overlapping physiological tolerance zones in humid monsoon alluvium.
2. **Asymmetric Agronomic Risk:** Under the conservative 4:1 cost penalty (where false recommendation carries 4x the cost of missed opportunity), **Extra Trees minimizes aggregate farmer penalty ($17,115.0$)**, outperforming HistGBM ($17,390.0$) and drastically beating KNN ($24,380.0$).
3. **Statistical Indistinguishability:** Top models are statistically tied ($p = 0.829$), making inference latency, explainability, and risk penalty the decisive evaluation criteria.
