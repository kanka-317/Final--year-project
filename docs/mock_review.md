# Mock Peer Review Report (Agent 11 - Adversarial Reviewer)

**Manuscript Title:** Integrating Soil Fertility, Climate and Geospatial Variables for Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal, India  
**Review Type:** Rigorous Adversarial Pre-Submission Peer Review  
**Date:** 2026-10-24

---

## 1. Overall Evaluation
The manuscript presents an extensive empirical evaluation of machine learning models for agricultural decision support across West Bengal. The authors address a critical real-world problem: integrating soil chemical tests, meso-climate, and geospatial variables for multi-crop suitability and fertilizer recommendation. However, several critical methodological and presentation aspects require rigorous defense before acceptance in high-impact journals (e.g., *Computers and Electronics in Agriculture* or *Geoderma*).

---

## 2. Major Reviewer Concerns & Comments

### Reviewer Comment 1: Multi-Class Accuracy Ceiling (~43%)
- **Concern:** The reported maximum multi-class test accuracy is $42.95\%$ for Extra Trees Classifier. In standard computer science benchmarks, $43\%$ accuracy might be considered low. How do the authors justify that this model is ready for agricultural decision support?
- **Required Author Action:** Quantify whether this represents a methodological deficiency or an inherent biological/survey characteristic. Compare against the theoretical random baseline ($8.33\%$).

### Reviewer Comment 2: Data Leakage Verification
- **Concern:** Agricultural datasets are notoriously susceptible to spatial autocorrelation and preprocessing leakage. If scalers or encoders were fitted across the entire dataset, the reported metrics are overly optimistic.
- **Required Author Action:** Provide proof that transformers were fitted strictly within training folds, and present a spatial cross-validation comparison (e.g., GroupKFold by district) to prove models do not merely memorize spatial coordinates.

### Reviewer Comment 3: Micronutrient Modeling Claims
- **Concern:** Many recent papers claim that machine learning can predict complete soil nutrient profiles from satellite remote sensing. Does your 12-parameter error analysis support or refute this?
- **Required Author Action:** Provide clear empirical metrics for each of the 12 soil parameters. Be candid if certain parameters fail to show predictability.

### Reviewer Comment 4: Statistical Significance of Top Models
- **Concern:** The authors test over 20 algorithms. Claiming Extra Trees is "better" than Random Forest ($42.15\%$) or HistGBM ($42.03\%$) based on minor decimal margins is statistically invalid without multi-comparison correction.
- **Required Author Action:** Conduct a formal non-parametric omnibus test (Friedman test) and pairwise tests with family-wise error rate correction (Holm or Nemenyi).

### Reviewer Comment 5: Synthetic Data Transparency
- **Concern:** If the dataset has synthetic components, this must be disclosed transparently to avoid academic misconduct.
- **Required Author Action:** Explicitly state the provenance of the prototyping dataset and verify reproducibility.
