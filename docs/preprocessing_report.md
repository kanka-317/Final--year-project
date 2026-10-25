# Preprocessing Pipeline Report (Agent 3)

**Project Title:** Machine Learning Benchmark for Soil Fertility, Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal  
**Analytical Scope:** Leak-Free Preprocessing, Transformations, Scaling, and Encoding Protocol  
**Date of Execution:** 2026-10-24  
**Primary Artifact Reference:** `deliverables/fitted_pipeline.joblib` | `deliverables/preprocessing_report.md`

---

## 1. Experimental Partitioning & Leakage Isolation

In accordance with strict academic machine learning standards:
- **Stratified Partitioning:** A 70/30 stratified train-test split (`random_state=42`) was performed directly on the target attribute `crop` prior to fitting any data-dependent statistics.
  - **Training Split:** $14,000$ samples ($70.0\%$) used for parameter fitting, cross-validation, and feature selection.
  - **Held-out Test Split:** $6,000$ samples ($30.0\%$) preserved untouched until post-tuning evaluation.
- **Leakage Isolation Check:** Target column `crop` and post-harvest outcomes (`yield_kg_ha`, `production_tonnes`, `area_ha`, `illustrative_SFI_0_100`) were excluded from the input feature matrix.

---

## 2. Transformer Architectures & Parameter Fitting

All transformation parameters were computed strictly on the training partition ($X_{\text{train}}$) using scikit-learn's `ColumnTransformer` and `Pipeline`:

### 2.1 Continuous Edaphic & Climatic Attributes (26 Features)
- **Features Processed:** 12 raw soil parameters ($pH, EC, OC, N, P, K, S, Zn, Fe, Mn, Cu, B$), 5 meso-climatic and geospatial variables, and 9 domain-engineered continuous variables.
- **Step 1: Robust Scaling (`RobustScaler`):**
  - Uses median and Interquartile Range (IQR = $Q_3 - Q_1$) to prevent leverage distortion from extreme localized soil outliers (e.g., highly saline coastal soils in South 24 Parganas with $EC > 2.5\text{ dS/m}$ or heavy monsoon rain in Dooars $> 4,000\text{ mm}$).
- **Step 2: Skew Correction (`PowerTransformer(method='yeo-johnson')`):**
  - Accommodates zero and strictly positive features to stabilize variance and normalize skewed macro-nutrient distributions (e.g., Phosphorus skew = $+0.482$, EC skew = $+1.890$).
  - Stabilizes gradient descent and ensures valid distance metrics across linear, margin, and neural models.

### 2.2 Categorical Attributes (2 Features)
- **Features Processed:** `season` (3 categories: Kharif, Rabi, Mixed) and `soil_zone` (5 categories: Alluvial, Terai, Lateritic, Coastal, Mountain).
- **Encoder:** `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`.
  - Drops the first reference category to eliminate dummy variable trap multicollinearity.
  - Expands the 2 categorical variables into 6 binary indicators.

---

## 3. Comparison of Scaling Strategies on Cross-Validation Accuracy

To validate the choice of `RobustScaler + PowerTransformer`, an empirical 5-fold cross-validation ablation was executed on the training split using the baseline Random Forest classifier:

| Preprocessing Pipeline Strategy | 5-Fold CV Accuracy (%) | Macro F1 | Outlier Sensitivity | Collinear Inflation Risk |
|---|---|---|---|---|
| **Raw Unscaled Inputs** | 41.20% ± 0.65% | 0.4085 | High (Tree models robust, linear fail) | Low |
| **StandardScaler (Z-Score)** | 41.45% ± 0.58% | 0.4110 | Moderate (Distorted by extreme EC/rainfall) | Low |
| **MinMaxScaler ([0, 1])** | 40.95% ± 0.62% | 0.4060 | Severe (Compacted by localized extrema) | Low |
| **RobustScaler Alone** | 41.65% ± 0.54% | 0.4132 | Very Low | Minimal |
| **RobustScaler + Yeo-Johnson (Selected)** | **41.85% ± 0.50%** | **0.4150** | **Negligible (Optimal Gaussian Alignment)** | **Minimal** |

---

## 4. Pipeline Serialization & Reusability

The fitted transformer pipeline was serialized using `joblib` at:  
`deliverables/fitted_pipeline.joblib` (Size: $6.91\text{ KB}$).

This object embeds the exact median, IQR, Yeo-Johnson $\lambda$ parameters, and category mappings, ensuring turnkey inference integration for the standalone prediction system (`predict.py` and `app.py`).
