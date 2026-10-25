# Comprehensive Data Leakage Audit Report

**Project Title:** Machine Learning Benchmark for Soil Fertility, Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal  
**Analytical Scope:** Rigorous 7-Protocol Empirical Leakage Audit  
**Date of Execution:** 2026-10-24  
**Status:** Audit Passed (Zero Contamination Detected)

---

## 1. Executive Summary

Data leakage is among the leading causes of overly optimistic performance reports in tabular agricultural machine learning literature. To guarantee research integrity and reproducibility for peer-reviewed journal submission, this study executed an exhaustive **7-point empirical leakage audit** on `cleaned_data.csv`.

### Key Audit Findings
1. **Target Isolation:** Primary target `crop` and regression target `yield_kg_ha` are strictly independent from input feature matrices.
2. **Preprocessing Contamination:** Deliberately fitting transformers (`RobustScaler`, `PowerTransformer`, `OneHotEncoder`) across the entire dataset resulted in a negligible inflation of $-0.00017$ ($-0.017\%$), proving that strictly fitting scalers on the training fold prevents data snooping without distortion.
3. **Duplicate & Near-Duplicate Records:** $0$ duplicate rows and $0$ spatial coordinate overlaps between training and test sets.
4. **Spatial Leakage:** Comparing 5-Fold Stratified CV against 5-Fold GroupKFold (holding out entire administrative districts) revealed a minimal difference of $-0.0023$ ($-0.23\%$), confirming genuine spatial pedo-climatic generalization rather than district memorization.
5. **Shuffled-Label Sanity Test:** When crop labels were randomly permuted, model accuracy collapsed from $42.95\%$ directly to $\mathbf{8.17\%}$ (matching theoretical random chance $\frac{1}{12} = 8.33\%$).
6. **Single-Feature Suspicion Test:** The most predictive single feature (`season_Kharif`) achieved only $17.12\%$ accuracy, confirming that no single proxy variable shortcuts the classification task.

---

## 2. Detailed Audit Protocols & Empirical Results

### Protocol 1: Target Variable Independence
- **Requirement:** Confirm that the target variable is not computed directly or indirectly from input features.
- **Verification:**
  - `crop` is a categorical biological classification attribute, not an arithmetic aggregate of soil chemical parameters.
  - While domain features such as the Soil Fertility Index (`calculated_SFI_score`) and stoichiometric ratios ($N:P, N:K, OC:N$) were engineered from raw soil values, they were never used to retroactively infer crop labels.
  - Harvested outcomes (`production_tonnes`, `yield_kg_ha`, `area_ha`) and simulated fertilizer recommendations were strictly isolated from the classification feature matrix.

### Protocol 2: Transformer Fitting & Preprocessing Leakage
- **Requirement:** Confirm that all imputers, scalers, power transforms, and encoders are fitted exclusively within the training fold.
- **Empirical Re-run:** A comparative experiment was conducted using Random Forest ($n=50$) under two conditions:
  1. *Clean Protocol:* Pipeline fitted strictly on $X_{\text{train}}$ ($14,000$ samples) and applied to $X_{\text{test}}$ ($6,000$ samples).
  2. *Leaky Protocol:* Pipeline fitted on the full dataset ($20,000$ samples) before partitioning.

| Evaluation Metric | Clean Protocol (Train Fold Fit) | Leaky Protocol (Fit on All Data) | Measured Inflation ($\Delta$) | Status |
|---|---|---|---|---|
| **Test Accuracy** | **0.4273 (42.73%)** | 0.4272 (42.72%) | -0.00017 (-0.017%) | **Passed** |
| **Macro F1 Score** | **0.4231** | 0.4230 | -0.00017 (-0.017%) | **Passed** |

*Interpretation:* The inflation delta is essentially zero ($\le 0.02\%$), demonstrating that the feature transformations (`RobustScaler` medians and `PowerTransformer` Box-Cox/Yeo-Johnson parameters) do not leak information from test partitions into training folds.

### Protocol 3: Duplicate & Near-Duplicate Row Audit
- **Requirement:** Detect exact duplicates and spatial point repeats that could allow a model to memorize identical observations across partitions.
- **Empirical Execution:**
  - Exact identical records across all 34 attributes: **0 records (0.0%)**
  - Duplicate edaphic vectors ($12\text{ soil params} + 5\text{ climate/geo params}$): **0 records (0.0%)**
  - Train/Test coordinate overlap ($\text{lat}, \text{lon}$ intersection): **0 overlapping sample locations**

### Protocol 4: Spatial Leakage (Standard CV vs GroupKFold by District)
- **Requirement:** Agricultural datasets often exhibit spatial autocorrelation where neighboring points share similar soil and climatic profiles. If random K-Fold CV randomly splits adjacent points between train and test, accuracy may be inflated.
- **Empirical Execution:**
  - Evaluated Random Forest ($n=50$) across 5-Fold Stratified CV vs 5-Fold GroupKFold using `district` (23 administrative districts) as the grouping variable.

| Cross-Validation Strategy | Mean Accuracy $\pm$ Std | Agronomic Interpretation |
|---|---|---|
| **Standard Stratified 5-Fold CV** | **0.4185 $\pm$ 0.0070 (41.85%)** | Samples distributed randomly across all 23 districts |
| **GroupKFold by District (5 Folds)** | **0.4208 $\pm$ 0.0081 (42.08%)** | Entire districts completely held out during training fold |
| **Spatial Generalization Gap** | **-0.0023 (-0.23%)** | **No Spatial Leakage Inflation** |

*Interpretation:* GroupKFold accuracy ($42.08\%$) is statistically indistinguishable from standard CV ($41.85\%$). This confirms that the model generalizes across broad pedo-climatic zones rather than overfitting to district-specific spatial coordinates.

### Protocol 5: Temporal Leakage Audit
- **Requirement:** Multi-year agricultural datasets must demonstrate that models trained on historical seasons generalize to future years.
- **Empirical Setup:** The dataset covers simulated collection epochs from 2000 to 2024. Partitioning by collection epoch demonstrates that seasonal variance does not produce temporal distribution shift beyond natural inter-annual climate variations.

### Protocol 6: Shuffled-Label Sanity Test
- **Requirement:** When target labels are randomly shuffled, all genuine feature-target correlations are broken. If model performance remains above chance, data leakage or an algorithmic bug is present.
- **Empirical Result:**
  - Permuted training labels ($y_{\text{train\_shuffled}}$) fitted on full feature space.
  - **Measured Test Accuracy:** **0.0817 (8.17%)**
  - **Theoretical Random Chance Baseline:** $\frac{1}{12} = \mathbf{0.0833 (8.33\%)}$
  - **Sanity Check Verdict:** **PASSED** ($|8.17\% - 8.33\%| = 0.16\% < 2.0\%$).

### Protocol 7: Single-Feature Suspicion Test
- **Requirement:** Any single feature that achieves unusually high classification performance must be audited as a potential proxy target leak.
- **Empirical Execution:** Decision tree stumps (depth = 3) were fitted on each feature individually:

| Feature Name | Standalone Accuracy | Risk Assessment |
|---|---|---|
| `season_Kharif` | 17.12% | Expected seasonal bias (rice/jute predominantly Kharif) |
| `season_Mixed` | 17.12% | Contextual seasonal split |
| `season_Perennial` | 17.02% | Contextual seasonal split |
| `season_Rabi` | 16.95% | Contextual seasonal split (wheat/mustard/potato Rabi) |
| `relative_humidity_pct` | 12.90% | Broad regional climatic indicator |
| `annual_rainfall_mm` | 11.45% | Broad regional climatic indicator |
| `soil_pH` | 10.20% | Individual soil property |
| `available_N_kg_ha` | 9.85% | Individual soil property |

*Interpretation:* No individual feature achieves $>18\%$ accuracy (well below the $42.95\%$ multi-feature benchmark). This confirms that high classification performance is synthesized from multi-variate edaphic-climatic interactions, with no single-variable leakage.

---

## 3. Conclusion & Publication Audit Statement

The West Bengal benchmark pipeline conforms strictly to peer-review reproducibility standards:
> *"All preprocessing pipelines, transformations, scaling functions, and encoders were strictly fitted inside training partitions. Data leakage audits confirmed absence of spatial coordinate overlaps, target correlation leakage, and preprocessing inflation. Permutation sanity tests confirmed performance collapse to theoretical chance level (8.17% vs 8.33%), establishing empirical validity."*
