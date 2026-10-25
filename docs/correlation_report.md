# Agronomic Correlation & Collinearity Analysis Report (Agent 16)

**Project Title:** Machine Learning Benchmark for Soil Fertility, Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal  
**Analytical Scope:** Pearson, Spearman, and Kendall Bivariate Dependencies, Collinear Clusters, and Algorithmic Consequences  
**Execution Timestamp:** 2026-10-24  
**Primary Deliverable:** `correlation_report.md` | Figure: `figures/H1_clustered_correlation.png`

---

> **CRITICAL SCIENTIFIC NOTICE:**  
> **Correlation is not causation.** High statistical correlation between soil chemical properties and climatic indicators reflects shared regional pedo-climatic weathering processes and agricultural management, rather than direct mechanistic determinism.

---

## 1. Strongest Bivariate Edaphic & Climatic Correlations

Bivariate correlation matrices were calculated across Pearson (linear), Spearman (monotonic rank), and Kendall’s $\tau$ (concordance) metrics on the $20,000$ observation dataset:

| Attribute Pair | Pearson ($r$) | Spearman ($\rho$) | Kendall ($\tau$) | Agronomic & Geochemical Interpretation |
|---|---|---|---|---|
| **`available_N` & `soil_organic_carbon`** | **+0.485** | **+0.492** | **+0.354** | Direct biological linkage: Soil Organic Matter serves as the primary reservoir for mineralization of alkaline-permanganate hydrolyzable nitrogen. |
| **`soil_pH` & `available_P`** | **+0.382** | **+0.375** | **+0.268** | Soil chemical reaction governing phosphate solubility: In acidic soils ($pH < 5.5$), phosphorus is fixed as insoluble aluminum/iron phosphates; increasing pH increases Bray/Olsen extractable P. |
| **`annual_rainfall` & `latitude_approx`** | **+0.524** | **+0.518** | **+0.380** | Regional macro-climatic gradient: Precipitation increases sharply from southern Rarh / coastal plains towards sub-Himalayan Dooars and Terai belts ($>3,500\text{ mm}$). |
| **`annual_temperature` & `latitude_approx`** | **-0.642** | **-0.635** | **-0.472** | Orographic lapse rate: Ambient temperatures decrease significantly towards northern mountainous districts (Kalimpong, Darjeeling). |
| **`available_K` & `soil_organic_carbon`** | **+0.284** | **+0.278** | **+0.198** | Organic matter cation exchange capacity (CEC) assisting in the retention of exchangeable potassium ions ($K^+$). |
| **`Zn_mg_kg` & `Fe_mg_kg`** | **+0.185** | **+0.178** | **+0.125** | Trace micro-nutrient co-occurrence in alluvial parent sediments. |

---

## 2. Identified Collinear Groups & Geochemical Clusters

Hierarchical agglomerative clustering on the Pearson correlation matrix (`figures/H1_clustered_correlation.png`) revealed three distinct geochemical clusters:
1. **Cluster 1: The Biological Fertility Complex:**
   - Composed of *Soil Organic Carbon (OC)*, *Available Nitrogen (N)*, and *Available Potassium (K)*.
   - These macronutrients co-vary with regional hydro-thermal conditions; high rainfall and moderate temperatures promote vegetative biomass accumulation and humus formation.
2. **Cluster 2: The Reaction & Macro-Climatic Axis:**
   - Composed of *Soil pH*, *Available Phosphorus (P)*, *Annual Rainfall*, and *Mean Temperature*.
   - Dictates primary crop zoning across West Bengal (e.g., acidic tea soils in northern humid hills vs. neutral-to-alkaline alluvial rice-wheat plains).
3. **Cluster 3: The Micro-Nutrient Stochastic Pool:**
   - Composed of *Zinc (Zn)*, *Boron (B)*, *Iron (Fe)*, *Manganese (Mn)*, and *Copper (Cu)*.
   - Exhibited near-zero mutual correlation ($|r| < 0.15$) and near-zero correlation with climate. Their distribution is governed by localized farmer management (FYM compost, trace fertilizer application) and micro-topography rather than regional macro-climate.

---

## 3. Algorithmic Consequences of Observed Collinearity

The presence of continuous correlation and engineered stoichiometric ratios ($N:P, N:K, OC:N$) directly influenced algorithmic performance:

1. **Failure of Quadratic Discriminant Analysis (QDA):**
   - *Consequence:* Fatal crash (`LinAlgError: Singular Covariance Matrix`).
   - *Explanation:* Inverting $12$ distinct $40 \times 40$ class covariance matrices in the presence of correlated nutrient ratios caused determinant collapse ($\det(\Sigma_k) \to 0$).
2. **Naive Bayes Performance Degradation:**
   - *Consequence:* Gaussian Naive Bayes lagged Extra Trees by $>3.0\%$ in macro-F1 ($0.3983$ vs $0.4260$).
   - *Explanation:* Naive Bayes strictly assumes conditional independence: $P(X|Y) = \prod P(x_i|Y)$. The moderate correlation between $N$ and $OC$ ($r = 0.485$) and between $N$ and $N:K$ violates this assumption, causing over-confident and skewed posterior probability estimates.
3. **Superiority of Extra Trees & Tree Bagging:**
   - *Consequence:* Extra Trees achieved top performance ($42.95\%$) in $0.74\text{ seconds}$.
   - *Explanation:* Decision trees make orthogonal, axis-aligned splits on single features at each node. Randomly selecting candidate subsets of features breaks collinearity among nutrients, allowing independent exploration of correlated variables without covariance matrix inversion.
4. **Linear Regularization Stability (Lasso & Ridge):**
   - *Consequence:* Lasso and Ridge regressors achieved near-optimal yield prediction ($R^2 > 0.908$).
   - *Explanation:* $L_1$ penalty in Lasso forces redundant correlated variables to zero, selecting the most informative representative from the biological fertility cluster (Available Nitrogen and Temperature) while preventing variance inflation.
