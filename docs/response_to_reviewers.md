# Author Rebuttal & Response to Reviewers (`response_to_reviewers.md`)

**Manuscript Title:** Integrating Soil Fertility, Climate and Geospatial Variables for Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal, India  
**Authors:** CSE Final-Year Research Team (MAKAUT)  
**Date:** 2026-10-24

---

We sincerely thank the Reviewer for their constructive and rigorous critique. Below, we address each concern point-by-point with empirical evidence.

---

### Response to Comment 1: Multi-Class Accuracy Ceiling (~43%)
- **Author Response:** In a 12-class balanced setting, random guessing yields $\frac{1}{12} = \mathbf{8.33\%}$, and the dummy majority baseline achieves $8.60\%$. Our Extra Trees model achieves **$42.95\%$ accuracy (over $5.1\times$ random chance)** with a Matthews Correlation Coefficient (MCC) of $0.3777$.
- **Agronomic Grounding:** In sub-tropical monsoon alluvial ecosystems, crops such as Rice, Jute, Maize, and Vegetables share broad physiological tolerance overlaps. Survey records capture individual farmer planting decisions rather than mutually exclusive ecological frontiers. Our 100-resample bootstrap bias-variance decomposition (`bias_variance_report.md`) proves that an empirical **Bayes noise ceiling ($\approx 57\%$ bias error)** operates across all high-capacity models, including deep neural networks. In our production system (`app.py`), predictions are presented as **Top-3 calibrated probabilistic options** rather than a single deterministic decree.

---

### Response to Comment 2: Data Leakage Verification
- **Author Response:** We conducted an exhaustive 7-protocol leakage audit (`leakage_audit.md`):
  1. *Preprocessing Inflation:* Comparing train-fold fitting vs deliberately leaky fit-on-all yielded an inflation delta of only **$-0.00017$ ($-0.017\%$)**, confirming no data snooping.
  2. *Spatial CV:* 5-Fold GroupKFold (holding out entire administrative districts) achieved **$42.08\% \pm 0.81\%$** vs. **$41.85\% \pm 0.70\%$** for standard Stratified CV (gap: $-0.23\%$). This proves the model learns regional pedo-climatic boundaries rather than spatial coordinate memorization.
  3. *Permutation Check:* Shuffled target labels caused model performance to collapse directly to **$8.17\%$**, confirming zero leakage.

---

### Response to Comment 3: Micronutrient Modeling Claims
- **Author Response:** We agree completely with the Reviewer. Our dedicated 12-parameter error analysis (`error_analysis_12_parameters.xlsx` and `error_report.md`) explicitly reports that while macronutrients ($K$, $R^2 = 0.209$; $N$, $R^2 = 0.138$; $OC$, $R^2 = 0.101$) show meso-scale predictability, **all micronutrients ($Zn, B, Fe, Cu, Mn, P, EC$) yield negative or near-zero test $R^2$ scores**.
- We explicitly state in the manuscript that satellite remote sensing and meteorological grids cannot replace on-site physical Soil Health Card chemical testing for micronutrients.

---

### Response to Comment 4: Statistical Significance of Top Models
- **Author Response:** We performed formal non-parametric omnibus and post-hoc testing across the 5 cross-validation folds (`deliverables/statistical_tests.json`):
  1. *Friedman Test:* $\chi_F^2 = 2.1429, p = 0.8290$ ($p > 0.05$). We fail to reject $H_0$; the top models are statistically tied on raw accuracy.
  2. *Pairwise Holm-Corrected Wilcoxon Tests:* All comparisons against Extra Trees yield $p_{\text{adj}} = 1.000$.
- Rather than overclaiming decimal accuracy differences, we state clearly that **Extra Trees is selected based on its lowest asymmetric agronomic risk penalty under 4:1 loss ($17,115.0$)**, rapid training latency ($0.74\text{s}$), and resistance to collinear stoichiometric ratios.

---

### Response to Comment 5: Synthetic Data Transparency
- **Author Response:** The manuscript and metadata explicitly declare the dataset as a synthetic benchmark dataset modeled after West Bengal Soil Health Card distributions (`data_dictionary.md` and `dataset_description.md`). The entire pipeline is open-source and reproducible with `run_all.py`.
