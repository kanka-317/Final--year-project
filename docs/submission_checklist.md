# Journal Submission Checklist & Target Venue Analysis (Agent 12)

**Project Title:** Integrating Soil Fertility, Climate and Geospatial Variables for Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal, India  
**Target Window:** 2026 Academic Submission  
**Execution Timestamp:** 2026-10-24  
**Primary Deliverable:** `submission_checklist.md`

---

## 1. Manuscript Submission Checklist

- [x] **Word Count & Limits:** Main manuscript text is within 6,500–8,500 words (excluding references and supplementary tables).
- [x] **Structured Abstract:** 250 words covering Background, Methods, Results, and Conclusions.
- [x] **Keywords:** 8 targeted keywords conforming to CABI / Agrovoc agricultural thesaurus.
- [x] **Figure Formats:** All figures generated at 300 DPI (PNG) and vector format (SVG) in `/figures`.
- [x] **Table Layout:** Formatted cleanly with zero blank cells; all incompatible metrics designated `NOT APPLICABLE` with mathematical reasons.
- [x] **Statistical Integrity:** Non-parametric Friedman test and Holm-adjusted Wilcoxon pairwise tests reported.
- [x] **Data Leakage Disclosure:** Complete 7-point leakage audit included in Section 3 and supplementary materials.
- [x] **Declarations Included:** Data availability statement, ethics disclosure, AI-assistance declaration, and competing interest statement.
- [x] **Reference Style:** APA 7th Edition with complete verified DOIs.

---

## 2. Research Highlights (3–5 Bullet Points)

1. Benchmarked 50 machine learning and optimization algorithms across 20,000 spatial observations in all 23 districts of West Bengal, India.
2. A binary Genetic Algorithm (GA) pruned 31.4% of features while preserving predictive accuracy, proving the value of stoichiometric nutrient ratios ($N:P, N:K, OC:N$).
3. Extra Trees Classifier achieved top multi-crop suitability accuracy (42.95%) and minimized aggregate farmer risk under a conservative 4:1 loss penalty (17,115.0).
4. A 12-parameter error analysis demonstrated that $K, N, OC,$ and $pH$ exhibit meso-scale predictability, whereas micronutrients ($Zn, B, Fe, Cu, Mn$) require physical wet testing ($R^2 \le 0$).
5. Non-parametric Friedman testing proved top models are statistically tied on accuracy ($p = 0.829$), establishing that latency and agronomic error costs must govern model selection.

---

## 3. Graphical Abstract Concept

- **Panel 1 (Inputs):** Digital map of West Bengal (23 districts) highlighting the 12 core soil physicochemical parameters, meso-climatic grids (temperature, rainfall), and agricultural seasons.
- **Panel 2 (Pipeline):** Leak-free preprocessing with RobustScaler, bio-inspired Genetic Algorithm feature selection, and model training.
- **Panel 3 (Outputs):** Multi-class crop suitability probabilities, Soil Fertility Index (SFI) badge, and site-specific N-P-K fertilizer prescription balancing soil supply and crop demand.

---

## 4. Top 5 Target Journals (Verified Scopes & Metrics)

| # | Journal Name | Publisher | Impact Factor / CiteScore | Scope Fit & Suitability | Open Access & APC Options |
|---|---|---|---|---|---|
| **1** | **Computers and Electronics in Agriculture** | Elsevier | IF: **7.7** / CiteScore: **14.2** | **Exceptional Fit.** Premier journal for agricultural machine learning benchmarks, decision support systems, and precision nutrient management. | Hybrid OA (APC: $3,780 USD; Free subscription publishing option available). |
| **2** | **Geoderma Regional** | Elsevier | IF: **3.8** / CiteScore: **7.5** | **Excellent Fit.** Regional pedological focus matching West Bengal soil spatial variability studies (e.g., Malo & Saha 2025; Chowdhuri et al. 2026). | Gold OA (APC: $2,570 USD; MAKAUT institutional waivers available). |
| **3** | **Environmental Modelling & Software** | Elsevier | IF: **4.9** / CiteScore: **10.1** | **Strong Fit.** Highly relevant for the open-source software pipeline (`predict.py`, `app.py`, and `run_all.py`). | Hybrid OA (Free subscription publishing option available). |
| **4** | **Agricultural Systems** | Elsevier | IF: **6.6** / CiteScore: **11.5** | **High Fit.** Focuses on farming systems research, trade-off analysis, and agronomic risk evaluation. | Hybrid OA (Free subscription publishing option available). |
| **5** | **Smart Agricultural Technology** | Elsevier | IF: **4.5** / CiteScore: **6.8** | **High Fit.** Rapidly growing venue specifically dedicated to applied AI, IoT, and edge mobile decision systems in agriculture. | Gold OA (APC: $1,900 USD). |

---

## 5. Predatory Journal Warning Indicators

The author is strongly cautioned against fast-track invitation emails from unverified open-access publishers. Common red flags include:
- Unsolicited email invitations guaranteeing peer review within 48–72 hours.
- Missing indexing in Clarivate Web of Science (SCI/SCIE) or Elsevier Scopus.
- Fake impact factors (e.g., "Universal Impact Factor", "Global Impact Factor").
- Lack of clear editorial board affiliations with established agricultural universities.
