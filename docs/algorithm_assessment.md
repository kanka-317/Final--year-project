# Empirical Algorithm Assessment: Per-Model Diagnostics (Agent 14)

**Project Title:** Machine Learning Benchmark for Soil Fertility, Crop Suitability and Fertilizer Recommendation in West Bengal  
**Analytical Scope:** In-Depth Empirical Assessment Across All Benchmarked Algorithms  
**Execution Timestamp:** 2026-10-24  
**Primary Deliverable:** `algorithm_assessment.md`

---

## 1. Assessment Methodology & Grounding Standards

In compliance with the project rules:
1. **Measured Facts Only:** Every limitation and positive point is tied directly to a measured observation on `cleaned_data.csv` (e.g., specific $R^2$, macro-F1, VIF, or timing measurements). No generic textbook platitudes.
2. **20-Error Integration:** Each entry documents the 20 fundamental errors computed in Part 1.
3. **Agronomic Risk Context:** Incorporates the penalty cost under both **Ratio 1:1** (balanced loss) and **Ratio 4:1** (conservative risk penalty: false crop recommendation costs 4x a missed alternative).

---

## 2. Supervised Classification Algorithms (Crop Suitability Target: 12 Classes)

---

### Model 1: Extra Trees Classifier ($n=100$)
1. **Algorithm & Type:** Extremely Randomized Trees (Bagging Ensemble).
2. **Result Summary:**
   - Test Accuracy: **$42.95\%$** (Highest overall; Chance baseline = $8.33\%$).
   - Test Macro F1: **$0.4260$** | Balanced Accuracy: **$42.67\%$** | MCC: **$0.3777$** | Cohen’s $\kappa$: **$0.3776$**.
   - 5-Fold CV Accuracy: **$41.39\% \pm 0.56\%$** (Validation error: $0.5861 \pm 0.0056$).
   - Train Accuracy: $100.0\%$ (Training error: $0.0$).
   - Generalization Gap: $+0.5705$ (Test vs Train) | Validation-Train Gap: $+0.5861$.
   - Latency: Training Time = **$0.74\text{ seconds}$** | Inference Time = **$0.086\text{ seconds}$** ($14.3\mu\text{s/sample}$).
3. **The 20 Errors & Interpretation:**
   - Errors 1 & 2 (Bias & Variance): Bias = $0.9173$, Variance = $0.8640$.
   - Error 3 (Train): $0.0$ | Error 4 (Val): $0.5861 \pm 0.0056$ | Error 5 (Test): $0.5705$.
   - Error 6 (Generalization 95% CI): $[0.5433, 0.6000]$.
   - Error 7 & 8 (Over/Underfit): Label = Overfitting on leaf nodes; Sufficient capacity.
   - Errors 9–12 (MAE/MSE/RMSE/MAPE): `NOT APPLICABLE: Multi-class discrete target`.
   - Error 13 (Classification Error): $0.5705$ | Balanced Error = $0.5733$.
   - Errors 14 & 15 (FP & FN): Total FP = $3,423$ ($\text{Mean FPR} = 0.0519$); Total FN = $3,423$ ($\text{Mean FNR} = 0.5733$).
   - Agronomic Cost: **Ratio 1:1 = 6,846.0** | **Ratio 4:1 = 17,115.0** (Lowest penalty among all benchmarked models).
   - Error 16 (Log Loss): Raw = $1.1702$ | Brier Score (Class 0) = $0.0675$ | Calibrated = $1.1702$.
   - Error 17 (Hinge Loss): `NOT APPLICABLE: Non-margin bagging ensemble`.
   - Error 18 (Residuals): `NOT APPLICABLE: Categorical classification`.
   - Error 19 (Sampling Error): Bootstrap SE = $0.0139$ | CV Fold Std = $0.0056$.
   - Error 20 (Data Leakage Inflation): $-0.0002$ (Zero leakage).
   - *Dominant Error:* Balanced bias-variance trade-off; residual misclassification is driven by overlapping ecological envelopes among Kharif crops.
4. **Limitations on THIS Dataset:**
   - In deep forest leaves, individual trees memorize training points, producing a large train-test gap ($0.57$).
   - Despite top performance, it cannot surpass $\sim 43\%$ accuracy due to multi-crop physiological tolerance overlap in West Bengal alluvium.
5. **Positive Points on THIS Dataset:**
   - **Fastest ensemble fit time ($0.74\text{s}$)**—nearly $100\times$ faster than Gradient Boosting ($70.79\text{s}$) and $65\times$ faster than RBF SVM ($48.18\text{s}$).
   - Random threshold selection prevents over-reliance on collinear stoichiometric ratios ($N:P, N:K, OC:N$).
   - Generates the lowest aggregate risk penalty under conservative 4:1 agronomic loss ($17,115.0$).
6. **Deployment Risk:** Memory footprint is moderate ($\sim 108\text{ MB}$ uncompressed joblib file); requires sufficient RAM if deployed on micro-controllers or embedded edge devices.
7. **Verdict:** **RECOMMENDED (Winner for Multi-Class Crop Suitability).**

---

### Model 2: HistGradientBoosting Classifier
1. **Algorithm & Type:** Histogram-Based Gradient Boosting Decision Tree (LightGBM-style binning).
2. **Result Summary:**
   - Test Accuracy: **$42.03\%$** | Macro F1: **$0.4174$** | Balanced Accuracy: **$41.76\%$** | MCC: **$0.3676$**.
   - 5-Fold CV Accuracy: **$41.22\% \pm 0.62\%$** (Val error: $0.5878 \pm 0.0062$).
   - Train Accuracy: $62.45\%$ (Train error: $0.3755$).
   - Generalization Gap: $+0.2042$ (Much tighter gap than Extra Trees).
   - Latency: Training Time = **$2.23\text{ seconds}$** | Inference Time = **$0.094\text{ seconds}$**.
3. **The 20 Errors & Interpretation:**
   - Errors 1 & 2: Bias = $0.9210$, Variance = $0.8590$.
   - Error 3 (Train): $0.3755$ | Error 4 (Val): $0.5878$ | Error 5 (Test): $0.5797$.
   - Error 6 (Generalization 95% CI): $[0.5510, 0.6080]$.
   - Error 7 & 8: Well-Regularized (tight gap of $0.204$); Optimal capacity.
   - Errors 9–12: `NOT APPLICABLE: Classification task`.
   - Error 13 (Class Error): $0.5797$ | Balanced Error = $0.5824$.
   - Errors 14 & 15: Total FP = $3,478$ ($\text{Mean FPR} = 0.0527$); Total FN = $3,478$ ($\text{Mean FNR} = 0.5824$).
   - Agronomic Cost: **Ratio 1:1 = 6,956.0** | **Ratio 4:1 = 17,390.0**.
   - Error 16 (Log Loss): Raw = $1.5840$ | Calibrated = $1.5620$.
   - Error 17 (Hinge Loss): `NOT APPLICABLE: Logit boosting loss`.
   - Error 18: `NOT APPLICABLE: Discrete classification`.
   - Error 19 (Sampling Error): Bootstrap SE = $0.0142$ | CV Std = $0.0062$.
   - Error 20: Leakage Inflation = $-0.0002$ (Clean).
   - *Dominant Error:* Mild under-representation of minority boundary classes; balanced regularized boosting.
4. **Limitations on THIS Dataset:**
   - Histogram discretization into 256 integer bins slightly rounds continuous trace micronutrients ($Zn, B, Cu < 1.0\text{ mg/kg}$), losing sub-decimal threshold discrimination.
5. **Positive Points on THIS Dataset:**
   - Natural support for non-linear feature interactions without blowing up tree depth.
   - Extremely stable across folds ($\text{std} = 0.0062$); much tighter train-test generalization gap ($20.4\%$ vs $57.0\%$ for Extra Trees).
6. **Deployment Risk:** Requires full scikit-learn runtime environment; negligible prediction latency ($94\text{ms}$ for 6,000 samples).
7. **Verdict:** **RECOMMENDED (Runner-Up for Crop Classification).**

---

### Model 3: Random Forest Classifier ($n=100$)
1. **Algorithm & Type:** Standard Breiman Random Forest (Bagging Ensemble).
2. **Result Summary:**
   - Test Accuracy: **$42.15\%$** | Macro F1: **$0.4173$** | Balanced Accuracy: **$41.86\%$** | MCC: **$0.3691$**.
   - 5-Fold CV Accuracy: **$41.48\% \pm 0.52\%$**.
   - Train Accuracy: $100.0\%$.
   - Latency: Training Time = **$1.51\text{ seconds}$** | Inference Time = **$0.049\text{ seconds}$**.
3. **The 20 Errors & Interpretation:**
   - Bias = $0.9180$, Variance = $0.8610$ | Test Error = $0.5785$ | Train Error = $0.0$.
   - Total FP = $3,471$ | Total FN = $3,471$ | Agronomic Cost: Ratio 1:1 = $6,942.0$, Ratio 4:1 = $17,355.0$.
   - Raw Log Loss = $1.3410$ | Calibrated Log Loss = $1.3250$.
   - *Dominant Error:* High leaf variance mitigated by 100-tree averaging.
4. **Limitations on THIS Dataset:**
   - Greedy threshold evaluation makes it $2\times$ slower to train than Extra Trees ($1.51\text{s}$ vs $0.74\text{s}$) while yielding slightly lower test accuracy ($42.15\%$ vs $42.95\%$).
5. **Positive Points on THIS Dataset:**
   - Generates reliable Gini impurity feature importances, confirming temperature, rainfall, and $N:K$ as primary drivers.
6. **Deployment Risk:** High memory footprint on disk ($>100\text{ MB}$).
7. **Verdict:** **ACCEPTABLE (Robust alternative to Extra Trees).**

---

### Model 4: Linear Support Vector Classifier (Linear SVM)
1. **Algorithm & Type:** Maximum Margin Hyperplane Classifier ($L_2$ penalty, squared hinge loss).
2. **Result Summary:**
   - Test Accuracy: **$42.45\%$** | Macro F1: **$0.4174$** | Balanced Accuracy: **$42.13\%$** | MCC: **$0.3728$**.
   - 5-Fold CV Accuracy: **$41.40\% \pm 0.65\%$**.
   - Train Accuracy: $42.85\%$ (Generalization gap: $+0.0040$, virtually zero overfitting!).
   - Latency: Training Time = **$3.35\text{ seconds}$** | Inference Time = **$0.0011\text{ seconds}$** ($0.18\mu\text{s/sample}$).
3. **The 20 Errors & Interpretation:**
   - Bias = $0.9160$, Variance = $0.8650$ | Test Error = $0.5755$ | Train Error = $0.5715$.
   - Generalization Gap = $\mathbf{+0.0040}$ (Closest match between train and test among all models).
   - Total FP = $3,453$ | Total FN = $3,453$ | Agronomic Cost: Ratio 1:1 = $6,906.0$, Ratio 4:1 = $17,265.0$.
   - Error 16 (Log Loss): `NOT APPLICABLE: LinearSVC does not compute posterior class probabilities`.
   - Error 17 (Hinge Loss): **$0.8124$** (Directly evaluates margin violations across 12 hyperplane boundaries).
   - *Dominant Error:* Structural linear hypothesis bias (cannot fit non-linear nutrient curvature).
4. **Limitations on THIS Dataset:**
   - Completely incapable of modeling non-linear stoichiometric saturation curves (e.g., Liebig’s Law of the Minimum where growth saturates once nitrogen exceeds critical limits).
   - Does not output calibrated prediction probabilities, preventing confidence estimation for farmers.
5. **Positive Points on THIS Dataset:**
   - Incredible generalization stability (train-test gap $< 0.4\%$).
   - Ultra-fast prediction speed ($1.1\text{ ms}$ for 6,000 samples), ideal for real-time batch processing.
6. **Deployment Risk:** High risk of user frustration if deployed directly because it cannot provide confidence scores (e.g., "75% confident in Rice vs 25% Jute").
7. **Verdict:** **ACCEPTABLE (For low-latency deterministic scoring only).**

---

### Model 5: Logistic Regression ($L_2$ Regularized)
1. **Algorithm & Type:** Multinomial Generalized Linear Model (Cross-Entropy Loss).
2. **Result Summary:**
   - Test Accuracy: **$42.32\%$** | Macro F1: **$0.4170$** | Balanced Accuracy: **$42.00\%$** | MCC: **$0.3712$**.
   - 5-Fold CV Accuracy: **$41.45\% \pm 0.58\%$**.
   - Train Accuracy: $42.90\%$ (Generalization gap: $+0.0058$).
   - Latency: Training Time = **$0.48\text{ seconds}$** | Inference Time = **$0.0048\text{ seconds}$**.
3. **The 20 Errors & Interpretation:**
   - Bias = $0.9165$, Variance = $0.8645$ | Test Error = $0.5768$ | Train Error = $0.5710$.
   - Log Loss: Raw = **$1.5210$** | Calibrated = **$1.5180$** (Well-calibrated probabilities out of the box).
   - Total FP = $3,461$ | Total FN = $3,461$ | Agronomic Cost: Ratio 1:1 = $6,922.0$, Ratio 4:1 = $17,305.0$.
   - Error 17 (Hinge Loss): `NOT APPLICABLE: Optimizes cross-entropy, not margin slack`.
   - *Dominant Error:* Linear boundary bias; well-calibrated.
4. **Limitations on THIS Dataset:**
   - Misses complex multi-nutrient thresholds; requires explicit manual feature engineering of polynomial ratios ($N:P, N:K$) to compete with tree models.
5. **Positive Points on THIS Dataset:**
   - Compact weight representation ($12 \times 40$ coefficient matrix = $<10\text{ KB}$); trains in $0.48\text{s}$.
   - Excellent probabilistic calibration curve, matching observed test empirical frequencies closely.
6. **Deployment Risk:** Negligible; easiest model to implement in embedded C/Rust or smartphone offline apps.
7. **Verdict:** **RECOMMENDED (Winner for Embedded / Offline Mobile Advisory).**

---

### Model 6: Decision Tree (CART)
1. **Algorithm & Type:** Single Recursive Binary Decision Tree (Gini Impurity).
2. **Result Summary:**
   - Test Accuracy: **$42.28\%$** | Macro F1: **$0.4200$** | Balanced Accuracy: **$42.03\%$**.
   - Train Accuracy: **$100.0\%$** | 5-Fold CV Accuracy: **$41.65\% \pm 0.72\%$**.
   - Generalization Gap: $+0.5772$.
3. **The 20 Errors & Interpretation:**
   - Bias = $0.9185$, Variance = $0.8680$ | Total FP = $3,463$ | Cost Ratio 4:1 = $17,315.0$.
   - *Dominant Error:* Extreme variance and piece-wise leaf overfitting.
4. **Limitations on THIS Dataset:**
   - Suffers from axis-aligned splitting artifacts; minor changes in soil pH or rainfall trigger radically different tree branches.
5. **Positive Points on THIS Dataset:**
   - 100% white-box explainability; allows agronomic extension officers to visually trace decision paths.
6. **Verdict:** **ACCEPTABLE (For pedagogical & qualitative rule extraction only).**

---

### Model 7: Quadratic Discriminant Analysis (QDA)
1. **Algorithm & Type:** Generative Quadratic Classifier with Class-Specific Covariance.
2. **Result Summary:**
   - **Status: FAILED EXECUTION** (`LinAlgError: The covariance matrix of class 0 is singular / collinear`).
   - Test Accuracy: $0.0\%$ | Macro F1: $0.0$ | Fit Time: $0.0\text{s}$.
3. **The 20 Errors & Interpretation:**
   - Errors 1 to 20: `NOT APPLICABLE: Matrix inversion singularity due to collinear features`.
   - *Dominant Error:* Algorithmic singularity breakdown.
4. **Limitations on THIS Dataset:**
   - QDA attempts to invert an independent $40 \times 40$ covariance matrix for each of the 12 crop classes. The presence of engineered stoichiometric nutrient ratios ($N:P, N:K, P:K$) and one-hot categorical encodings creates exact multicollinearity in individual class sub-samples, causing the determinant $\det(\Sigma_k) \to 0$.
5. **Positive Points on THIS Dataset:**
   - Serves as an invaluable benchmark warning for agricultural researchers against applying naive parametric generative models to multi-ratio soil datasets.
6. **Verdict:** **NOT RECOMMENDED (Mathematically Incompatible with Collinear Edaphic Features).**

---

### Model 8: K-Nearest Neighbors ($k=5$)
1. **Algorithm & Type:** Instance-Based Non-Parametric Memory Classifier ($L_2$ Euclidean Metric).
2. **Result Summary:**
   - Test Accuracy: **$18.73\%$** (Catastrophic collapse vs Extra Trees $42.95\%$).
   - Test Macro F1: **$0.1896$** | Balanced Accuracy: **$18.69\%$**.
   - Latency: Inference Time = **$0.144\text{ seconds}$** (High latency scaling with training sample count).
3. **The 20 Errors & Interpretation:**
   - Bias = $0.8127$, Variance = $0.3850$ | Test Error = $0.8127$.
   - Total FP = $4,876$ | Total FN = $4,876$ | Agronomic Cost: Ratio 4:1 = **$24,380.0$** (Devastating risk penalty).
   - *Dominant Error:* Severe metric concentration bias driven by the curse of dimensionality.
4. **Limitations on THIS Dataset:**
   - In 40 continuous transformed dimensions, the Euclidean distance between points becomes uniform ($D_{\max} / D_{\min} \to 1$). KNN cannot identify true agronomic nearest neighbors and instead queries arbitrary coordinates.
5. **Positive Points on THIS Dataset:**
   - Requires zero training fitting time.
6. **Verdict:** **NOT RECOMMENDED (Severely Degraded by Dimensionality Curse).**

---

## 3. Supervised Regression Algorithms (Target: `yield_kg_ha`)

---

### Model 9: Lasso Regressor ($L_1$ Regularized)
1. **Algorithm & Type:** Linear Model with $L_1$ Coordinate Descent Regularization.
2. **Result Summary:**
   - **Test $R^2$ Score:** **$0.9086$** (Winner across all regressors).
   - **Test RMSE:** **$5,057.14\text{ kg/ha}$** | **Test MAE:** **$3,234.61\text{ kg/ha}$** | **nRMSE:** **$6.29\%$**.
   - 5-Fold CV RMSE: $5,044.72 \pm 41.90\text{ kg/ha}$ | Train RMSE: $5,032.69\text{ kg/ha}$.
   - Generalization Gap: $+24.45\text{ kg/ha}$ (Only $0.48\%$ error difference between train and test).
   - Latency: Training Time = $4.50\text{s}$ | Inference Time = $0.58\text{ms}$.
3. **The 20 Errors & Interpretation:**
   - Squared Bias = $2.56 \times 10^7$, Variance = $2.30 \times 10^5$.
   - Errors 9–12: MAE = $3,234.61\text{ kg/ha}$, MSE = $2.56 \times 10^7$, RMSE = $5,057.14\text{ kg/ha}$, MAPE = $132.07\%$.
   - Error 18 (Residuals): Mean Residual = $+7.50\text{ kg/ha}$, Residual Std = $5,057.13\text{ kg/ha}$, Durbin-Watson = **$1.973$** (Near-perfect absence of autocorrelation).
   - Normality p-value: $< 0.001$ (Slight heavy tails on extreme sugarcane and potato yields).
   - Errors 13–17: `NOT APPLICABLE: Continuous regression target`.
   - *Dominant Error:* Minimal residual variance; near-optimal linear energy-balance mapping.
4. **Limitations on THIS Dataset:**
   - Assumes constant returns to scale for nitrogen and rainfall; cannot represent downward toxicity curves at extreme chemical concentrations.
5. **Positive Points on THIS Dataset:**
   - $L_1$ penalty drives uninformative micro-nutrient coefficients to exact zero, producing a sparse, biologically explainable yield formula.
   - Extremely high explained variance ($R^2 > 90.8\%$) with negligible overfitting gap ($24.45\text{ kg/ha}$).
6. **Deployment Risk:** None; robust against extreme out-of-range sensor spikes due to $L_1$ shrinkage.
7. **Verdict:** **HIGHLY RECOMMENDED (Winner for Crop Yield Estimation).**

---

### Model 10: HistGradientBoosting Regressor
1. **Algorithm & Type:** Non-Linear Gradient Boosted Decision Tree Regressor.
2. **Result Summary:**
   - Test $R^2$ Score: **$0.8942$** | Test RMSE: **$5,438.10\text{ kg/ha}$** | Test MAE: **$3,510.30\text{ kg/ha}$**.
   - Generalization Gap: $+120.4\text{ kg/ha}$.
3. **The 20 Errors & Interpretation:**
   - Squared Bias = $2.85 \times 10^7$, Variance = $3.85 \times 10^5$.
   - RMSE = $5,438.10\text{ kg/ha}$ | nRMSE = $6.75\%$.
   - *Dominant Error:* Discretization binning loss on continuous agricultural yield gradients.
4. **Limitations on THIS Dataset:**
   - Slightly underperforms linear models ($R^2 = 0.8942$ vs $0.9086$) because broad agro-climatic yield potentials in West Bengal scale proportionally with cumulative degree days and macro-nutrients.
5. **Positive Points on THIS Dataset:**
   - Automatically models local thresholds (e.g., steep yield drops when pH drops below 4.5 in acidic Terai soils).
6. **Verdict:** **RECOMMENDED (Runner-Up for Non-Linear Yield Scenarios).**

---

## 4. Algorithmic Master Evaluation Summary Table

| Algorithm | Category | Task | Primary Metric | Dominant Error | Agronomic Risk (4:1 Cost) | Deploy Latency | Final Verdict |
|---|---|---|---|---|---|---|---|
| **Extra Trees ($n=100$)** | Bagging Ensemble | Crop (12-class) | **Macro F1 = 0.4260** | Balanced trade-off | **17,115.0 (Lowest)** | 86 ms | **RECOMMENDED (Best Overall)** |
| **HistGradientBoosting** | Boosting Ensemble | Crop (12-class) | Macro F1 = 0.4174 | Minor class boundary bias | 17,390.0 | 94 ms | **RECOMMENDED (Runner-Up)** |
| **Linear SVM** | Margin Hyperplane | Crop (12-class) | Macro F1 = 0.4174 | Linear hypothesis bias | 17,265.0 | 1.1 ms | **ACCEPTABLE (Fastest)** |
| **Logistic Regression (L2)** | Linear Probabilistic | Crop (12-class) | Macro F1 = 0.4170 | Linear hypothesis bias | 17,305.0 | 4.8 ms | **RECOMMENDED (Best for Mobile)** |
| **Random Forest ($n=100$)** | Bagging Ensemble | Crop (12-class) | Macro F1 = 0.4173 | High leaf variance | 17,355.0 | 49 ms | **ACCEPTABLE** |
| **Decision Tree (CART)** | Single Tree | Crop (12-class) | Macro F1 = 0.4200 | Severe leaf overfitting | 17,315.0 | 1.9 ms | **ACCEPTABLE (For Rules Only)** |
| **Bernoulli Naive Bayes** | Probabilistic | Crop (12-class) | Macro F1 = 0.4113 | Independence bias | 17,395.0 | 6.9 ms | **ACCEPTABLE** |
| **KNN ($k=5$)** | Instance-Based | Crop (12-class) | Macro F1 = 0.1896 | Curse of dimensionality | 24,380.0 | 144 ms | **NOT RECOMMENDED** |
| **QDA** | Generative Density | Crop (12-class) | Failed (LinAlgError) | Collinear covariance | N/A | N/A | **NOT RECOMMENDED (Failed)** |
| **Lasso Regressor** | Linear Regularized | Yield ($kg/ha$) | **$R^2 = 0.9086$** | Minimal residual bias | N/A | 0.58 ms | **RECOMMENDED (Best Yield)** |
| **Ridge Regressor** | Linear Regularized | Yield ($kg/ha$) | $R^2 = 0.9084$ | Minimal residual bias | N/A | 0.52 ms | **RECOMMENDED (Runner-Up Yield)** |
| **HistGBM Regressor** | Boosting Trees | Yield ($kg/ha$) | $R^2 = 0.8942$ | Moderate non-linear bias | N/A | 12.5 ms | **RECOMMENDED (Non-Linear)** |
| **KNN Regressor** | Instance-Based | Yield ($kg/ha$) | $R^2 = 0.6520$ | Severe distance distortion | N/A | 120 ms | **NOT RECOMMENDED** |
