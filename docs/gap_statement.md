# Research Gap Analysis and Formal Objectives

## 1. Grouped Syntheses of Regional Literature (2022–2026, West Bengal Focus)

### A. Soil Fertility Index (SFI) and Micronutrient Modeling
Recent benchmark investigations across West Bengal (notably Malo & Saha, 2025 in the Barind Tract of Dakshin Dinajpur, and Mullick et al., 2024 in South 24 Parganas) emphasize that assessing soil fertility requires moving beyond traditional macro-nutrients (N, P, K) to a 12-parameter composite model incorporating pH, EC, OC, S, and essential micronutrients (Fe, Cu, Zn, B, Mn). However, current regional implementations rely predominantly on classical Soil Nutrient Index (SNI) heuristics or static Min-Max weighted indices mapped via deterministic Inverse Distance Weighting (IDW). These approaches do not capture high-dimensional non-linear interactions between edaphic properties and fail to quantify prediction uncertainty across heterogeneous agro-ecological zones.

### B. Crop Suitability Frameworks
Contemporary state studies (Bose & Halder, 2023 at statewide resolution; Singha et al., 2023 in Tarakeswar Block; Chowdhuri et al., 2026 in Bankura) demonstrate the progressive shift from FAO land evaluation to multi-criteria decision making (AHP / Fuzzy-AHP) and early machine learning baselines (SVM, Logistic Regression). While these works highlight regional diversity—from the floodplains of Hooghly to the drought-prone lateritic fringes of Bankura—they remain constrained:
1. Suitability models evaluate only 1 to 2 focal crops (e.g., rice and potato) or broad crop groups without probabilistic multi-class ranking.
2. The models lack rigorous comparisons across modern ensemble algorithms (XGBoost, LightGBM, CatBoost) and bio-inspired optimizers.
3. Feature spaces omit explicit geochemical ratios (N:P, N:K, OC:N) and microclimate stress indices.

### C. Site-Specific Fertilizer Recommendation (SSNM)
Current agronomic recommendations (such as FertiliCalc used by Malo & Saha, 2025) operate via stoichiometric lookup tables subtracting baseline soil supply from crop removal targets. No published West Bengal study has systematically integrated dynamic ML-based multi-target regression to simultaneously model fertilizer demand (N, P, K) alongside crop suitability and continuous SFI.

### D. Climate, Topography, and Machine Learning Integration
Topographic catenary analysis (Chowdhuri et al., 2026) proves that agricultural potential in West Bengal is strongly mediated by toposequence and microclimate. However, existing literature treats GIS multi-criteria overlay and machine learning as disparate silos rather than an end-to-end reproducible pipeline with strict leakage prevention, nested cross-validation, and bio-inspired feature selection.

---

## 2. Defensible Research Gap Statement

> **Research Gap:** While individual components of soil fertility assessment (SNI/SFI), geospatial land evaluation (AHP/GIS), and agronomic nutrient calculations have been investigated across distinct blocks of West Bengal, there is currently **no unified, end-to-end computational framework that integrates the full suite of 12 soil physicochemical parameters with meso-climate and geospatial variables to simultaneously:**
> 1. Formulate predictive benchmarks across supervised, unsupervised, and contextual-bandit reinforcement learning paradigms.
> 2. Implement bio-inspired metaheuristic feature selection (e.g., Genetic Algorithms) to discover minimal non-redundant agro-edaphic feature subsets.
> 3. Provide site-specific multi-crop suitability rankings alongside calibrated, quantitative N-P-K fertilizer demand estimations with rigorous out-of-distribution uncertainty bounds.

---

## 3. Core Research Objectives

1. **Objective 1 (Multi-Parametric Edaphic & Climatic Benchmarking):**
   To benchmark a comprehensive catalogue of supervised (linear, tree-ensembles, deep neural), unsupervised (clustering and manifold learning), and nature-inspired algorithms on 12 soil parameters plus climate-geospatial attributes, establishing verified error baselines across all 12 parameters (RMSE, MAE, R², MAPE, normalized RMSE).

2. **Objective 2 (Bio-Inspired Feature Selection & Dimensionality Discovery):**
   To evaluate nature-inspired metaheuristics (Genetic Algorithm, Particle Swarm Optimization) against classical filter, wrapper (RFE), and embedded techniques, determining whether stoichiometric ratios (N:P, N:K, P:K, OC:N) and pH deviation metrics enhance classifier discrimination while minimizing multicollinearity.

3. **Objective 3 (Algorithmic Suitability & Fertilizer Quantification):**
   To formulate a multi-task recommendation architecture predicting top-3 crop suitability ranking with calibrated confidence, continuous Soil Fertility Index (SFI), and site-specific N-P-K fertilizer requirements.

4. **Objective 4 (Deployment System & Decision Support):**
   To implement a modular, leak-free preprocessing pipeline and interactive prediction interface supporting point coordinates, manual 12-parameter input, and batch CSV inference with explicit out-of-distribution uncertainty detection.
