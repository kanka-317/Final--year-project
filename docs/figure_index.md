# Scientific Figure Index & Visual Artifact Catalogue (Agent 16)

**Project Title:** Machine Learning Benchmark for Soil Fertility, Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal  
**Analytical Scope:** Complete Visual Asset Registry (300 DPI PNG & Vector SVG Formats)  
**Execution Timestamp:** 2026-10-24  
**Primary Directory Reference:** `figures/` and `figures/svg/`

---

## 1. Master Figure Registry

| Figure ID | File Name | Target Analytical Use | Format & Resolution | Key Agronomic & Statistical Interpretation |
|---|---|---|---|---|
| **Fig 1** | `H1_clustered_correlation.png` | Collinearity & Edaphic Groupings | 300 DPI PNG + SVG | Hierarchically clustered Pearson correlation heatmap revealing stoichiometric clusters ($N$ with $OC$; $P$ with $pH$; micro-nutrients as independent stochastic clusters). |
| **Fig 2** | `H2_vif_multicollinearity.png` | Multicollinearity Audit | 300 DPI PNG + SVG | Variance Inflation Factor (VIF) matrix confirming all raw soil and climate features reside between $1.00$ and $2.30$, well below the collinear breakdown threshold of $5.0$. |
| **Fig 3** | `H4_error_master_heatmap.png` | Unified Algorithmic Error Comparison | 300 DPI PNG + SVG | Column-normalized cross-algorithm comparison of Train, Validation, Test, Bias, and Variance errors, highlighting Extra Trees' superior balance. |
| **Fig 4** | `H5_12_parameter_error_heatmap.png`| 12 Soil Parameter Predictability | 300 DPI PNG + SVG | Cross-model $R^2$ heatmap illustrating the predictability dichotomy: $K$ ($0.209$), $N$ ($0.138$), $OC$ ($0.101$), and $pH$ ($0.098$) are predictable, whereas micronutrients ($Zn, B, Fe, Cu, Mn$) exhibit $R^2 \le 0$. |
| **Fig 5** | `H6_train_val_test_gap_heatmap.png`| Overfitting / Underfitting Diagnosis | 300 DPI PNG + SVG | Visualizes the generalization gap across models; contrasts tree leaf memorization with linear/HistGBM stability. |
| **Fig 6** | `H7_bias_variance_heatmap.png` | Bias-Variance Trade-Off | 300 DPI PNG + SVG | Bootstrap 0-1 loss decomposition showing that Bayes noise ($57\%$) constitutes the dominant error source across all high-capacity models. |
| **Fig 7** | `H12_pairwise_significance_heatmap.png`| Statistical Significance | 300 DPI PNG + SVG | Pairwise Holm-adjusted Wilcoxon signed-rank test $p$-values confirming top models (Extra Trees, HistGBM, RF, Linear SVM, LogReg) are statistically tied ($p = 1.000$). |
| **Fig 8** | `H14_decision_matrix_heatmap.png` | Multi-Criteria Recommendation | 300 DPI PNG + SVG | Weighted multi-criteria decision matrix scoring models across 10 dimensions, showing Extra Trees winning on accuracy/risk and LogReg winning on mobile edge deployment. |
| **Fig 9** | `H15_district_variation_heatmap.png`| Spatial Agro-Ecological Profiling | 300 DPI PNG + SVG | Standardized edaphic-climatic profiles across all 23 districts of West Bengal, capturing regional soil zones (Lateritic plateau, Gangetic alluvium, coastal delta, Terai). |
| **Fig 10** | `H16_feature_selection_ablation_heatmap.png`| Bio-Inspired Feature Ablation | 300 DPI PNG + SVG | Performance comparison showing GA feature selection ($24$ features) beating raw soil ($12$ features) and full engineered pool ($35$ features) while reducing latency by $22.8\%$. |
| **Fig 11** | `bias_variance_decomposition.png`| Bias vs Variance Bar Chart | 300 DPI PNG + SVG | Dual bar plot illustrating the bias-variance breakdown for top classifiers. |
| **Fig 12** | `cost_ratio_comparison.png` | Agronomic Cost Curves | 300 DPI PNG + SVG | Compares total weighted penalty costs under Ratio 1:1 vs. conservative Ratio 4:1 (Extra Trees maintains lowest farmer risk). |
| **Fig 13** | `confusion_matrices_fp_fn.png` | Error Confusion Matrix | 300 DPI PNG + SVG | Annotated heatmap of Extra Trees predictions highlighting off-diagonal False Positives and False Negatives. |
| **Fig 14** | `residual_plots.png` | Residual Diagnostics (Yield) | 300 DPI PNG + SVG | Homoscedasticity scatter plot and Normal Q-Q plot for Lasso yield regression (Durbin-Watson = $1.973$). |
| **Fig 15** | `calibration_curves.png` | Reliability Calibration Curves | 300 DPI PNG + SVG | Reliability diagrams across top probabilistic classifiers against the ideal 45-degree calibration diagonal. |
| **Fig 16** | `learning_curves.png` | Sample Size Scalability | 300 DPI PNG + SVG | Training and validation accuracy curves as sample size scales from 1,400 to 14,000 observations. |
| **Fig 17** | `top_feature_importance.png` | Gini Feature Importances | 300 DPI PNG + SVG | Top 15 influential variables driving multi-crop suitability (ambient temperature, annual rainfall, and $N:K$ stoichiometric ratio dominate). |
