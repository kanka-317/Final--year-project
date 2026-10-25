# Top 5 Figures for Manuscript Submission (`figure_summary.md`)

**Target Manuscript:** Integrating Soil Fertility, Climate and Geospatial Variables for Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal, India  
**Journal Target:** Computers and Electronics in Agriculture / Geoderma / Environmental Modelling & Software  
**Document Purpose:** Curated selection of the 5 indispensable visual exhibits for the main research text.

---

### 1. Figure 1: The 12 Soil Parameter Predictability Matrix (`H5_12_parameter_error_heatmap.png`)
- **Why It Is Crucial:** Directly resolves a primary scientific question in modern digital soil mapping: *Can point-of-care laboratory soil tests be replaced by satellite and meteorological machine learning models?*
- **What It Proves:** Shows a stark dichotomy. Potassium ($R^2 = 0.209$), Nitrogen ($R^2 = 0.138$), and Organic Carbon ($R^2 = 0.101$) exhibit meaningful meso-scale predictability. In contrast, micronutrients ($Zn, B, Fe, Cu, Mn$) yield negative $R^2$ across all models, proving empirically that physical wet-chemistry testing remains indispensable for micronutrient management.

### 2. Figure 2: Agronomic Recommendation Risk Penalty Curves (`cost_ratio_comparison.png`)
- **Why It Is Crucial:** Transitions machine learning from abstract mathematical accuracy to real-world agricultural economics.
- **What It Proves:** Demonstrates model behavior under shifting risk aversion: False Positive (recommending a crop that fails, causing debt) vs. False Negative (missing an alternative). Extra Trees minimizes aggregate risk penalty under both Ratio 1:1 ($6,846.0$) and conservative Ratio 4:1 ($17,115.0$), avoiding $1,453$ catastrophic misrecommendations compared to KNN.

### 3. Figure 3: Empirical Bias-Variance Decomposition (`bias_variance_decomposition.png` / `H7_bias_variance_heatmap.png`)
- **Why It Is Crucial:** Provides foundational theoretical justification for algorithmic performance on tabular agronomic data.
- **What It Proves:** Establishes that all top classifiers reach an empirical Bayes noise ceiling around $\sim 57\%$ bias error, driven by overlapping ecological tolerance envelopes in sub-tropical monsoon climates. It visually explains why Extra Trees wins (suppressing split variance without inflating bias).

### 4. Figure 4: Pairwise Statistical Significance Matrix (`H12_pairwise_significance_heatmap.png`)
- **Why It Is Crucial:** Essential for peer-reviewed rigor; prevents overclaiming minor accuracy decimal differences.
- **What It Proves:** Illustrates the non-parametric Friedman test ($\chi_F^2 = 2.14, p = 0.829$) and Holm-adjusted Wilcoxon pairwise tests ($p = 1.000$), showing that the top 6 models are statistically tied. This formally justifies selecting Extra Trees based on inference latency ($0.74\text{s}$) and agronomic risk, rather than insignificant accuracy increments.

### 5. Figure 5: Clustered Pearson Correlation Matrix (`H1_clustered_correlation.png`)
- **Why It Is Crucial:** Maps the geochemical interactions governing West Bengal soils.
- **What It Proves:** Uncovers natural stoichiometric groupings (available $N$ clustered with organic carbon; available $P$ clustered with soil reaction $pH$), explaining why bio-inspired Genetic Algorithms pruned $31.4\%$ of redundant features while preserving predictive accuracy.
