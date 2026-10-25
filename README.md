# West Bengal Soil Fertility, Crop Suitability & Site-Specific Fertilizer Recommendation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

An end-to-end Machine Learning research and precision agricultural intelligence system integrating soil fertility indices (SFI), agro-climatic variables, and geospatial coordinates across all 23 districts of West Bengal, India.

---

## 📌 Project Highlights

- **Master Dataset:** 20,000 field soil samples across 6 agro-climatic zones of West Bengal (Alluvial, Terai, Lateritic, Coastal, Mountain, Red-Laterite).
- **Proposed Novel Algorithm:** **`AgroStack-CR`** (Cost-Sensitive Agronomic Stacking Ensemble with Conformal Risk Rejection).
  - **Tripartite Orthogonal Stacking:** Blends Extra Trees (bagging to suppress collinear ratio sensitivity), HistGradientBoosting (boosting to capture non-linear thresholds), and Multinomial Logistic Regression ($L_2$ regularized linear prior).
  - **Temperature Probability Calibration ($T^* = 0.7807$):** Eliminates leaf-node overconfidence, achieving lowest Log Loss ($1.1316$) and Brier score ($0.0667$).
  - **Bayesian Minimum Expected Cost (MEC) Decoding:** Optimizes crop selection under an asymmetric 4:1 smallholder debt loss matrix ($C_{FP} = 4.0, C_{FN} = 1.0$).
  - **Split-Conformal Prediction Sets ($1 - \alpha = 0.90$):** Guarantees **$90.32\%$ empirical marginal coverage**, returning a safe portfolio of $\approx 3$ viable crops instead of a brittle single guess.
  - **Agronomic Rejection Gate ($\tau_{\text{reject}}$):** Flags $6.27\%$ of extreme or borderline queries for field soil re-testing rather than forcing high-risk crop failure.
- **Yield Regressor:** Lasso Regressor ($R^2 = 0.9086$, $\text{RMSE} = 5,057.14\text{ kg/ha}$).
- **Interactive Web App & CLI Tool:** Built with Streamlit (`app.py`) and standalone CLI (`predict.py`).

---

## 📂 Repository Directory Structure

```
├── app.py                     # Streamlit web application interface
├── predict.py                 # Standalone CLI prediction & advisory engine
├── cleaned_data.csv           # Master preprocessed agricultural dataset (20,000 samples)
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # Git ignore rules
│
├── notebooks/                 # Jupyter Research Notebooks
│   ├── part1_error_analysis_benchmark.ipynb
│   ├── part2_dataset_assessment_recommendations.ipynb
│   ├── part3_proposed_algorithm_agrostack_cr.ipynb
│   └── soil_crop_fertilizer_complete_benchmark.ipynb
│
├── deliverables/              # Trained Model Artifacts & Proposed Algorithm
│   ├── best_crop_model.joblib # Compressed Extra Trees baseline model
│   ├── fitted_pipeline.joblib # Leak-free preprocessing pipeline
│   ├── classifier_20_errors.json
│   ├── regressor_20_errors.json
│   ├── statistical_tests.json
│   └── proposed_algorithm/    # AgroStack-CR Implementation Package
│       ├── proposed_algorithm.py
│       ├── test_interactive.py
│       ├── proposed_algorithm_20_errors.json
│       ├── proposed_algorithm_error_analysis.docx
│       ├── proposed_algorithm_error_analysis.md
│       ├── algorithm_design.md
│       ├── diagnosis_report.md
│       ├── novelty_report.md
│       ├── pseudocode.md
│       ├── figures/           # Diagnostic charts (300 DPI)
│       └── tests/             # Pytest unit tests
│
├── manuscripts_and_reports/   # Microsoft Word (.docx) Academic Documents
│   ├── manuscript.docx
│   ├── proposed_algorithm_error_analysis.docx
│   ├── comprehensive_error_analysis_and_benchmark_report.docx
│   ├── algorithm_limitations_comprehensive_report.docx
│   ├── model_training_report.docx
│   ├── cover_letter.docx
│   └── supplementary.docx
│
├── docs/                      # 19 Technical Diagnostic Markdown Reports
│   ├── algorithm_assessment.md
│   ├── bias_variance_report.md
│   ├── leakage_audit.md
│   ├── dataset_description.md
│   └── references.bib
│
├── scripts/                   # Data processing and benchmark generator scripts
├── benchmark_data/            # Raw data spreadsheets and metric matrices
├── figures/                   # 16 Benchmark diagnostic plots and heatmaps
└── tests/                     # System deployment unit tests
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>
```

### 2. Install Dependencies
```bash
pip install -r deliverables/proposed_algorithm/requirements.txt
```

### 3. Run the Interactive Web Application
```bash
streamlit run app.py
```

### 4. Run CLI Soil Prediction
```bash
python predict.py --pH 6.2 --EC 0.45 --OC 0.65 --N 220 --P 24.5 --K 160 --S 18 --Zn 0.75 --Fe 18.5 --district Nadia --season Kharif
```

### 5. Run Unit Tests
```bash
pytest deliverables/proposed_algorithm/tests/ -v
```

---

## 📊 Benchmark Summary: The 20 Fundamental Errors

| Error Metric | AgroStack-CR (Proposed) | Extra Trees (Baseline) | HistGradientBoosting | Linear SVM |
|---|:---:|:---:|:---:|:---:|
| **Test Accuracy** | $41.80\%$ | $42.95\%$ | $42.03\%$ | $42.45\%$ |
| **Bias Error** | $0.5820$ | $0.5705$ | $0.5797$ | $0.5755$ |
| **Variance Error** | **$0.0003$ (Lowest)** | $0.0031$ | $0.0038$ | $0.0042$ |
| **Generalization Gap** | **$-0.0694$ (No Overfit)** | $+0.5705$ (Severe) | $+0.2042$ | $+0.0040$ |
| **Calibrated Log Loss** | **$1.1316$ (Best)** | $1.1702$ | $1.5840$ | N/A |
| **Brier Score** | **$0.0667$ (Best)** | $0.0675$ | $0.0712$ | N/A |
| **Agronomic Cost (4:1)** | $17,460.0$ | $17,115.0$ | $17,390.0$ | $17,265.0$ |
| **Conformal Coverage (90%)** | **$90.32\%$** | N/A | N/A | N/A |
| **Mean Conformal Set Size** | **$3.16$ crops** | N/A | N/A | N/A |

---

## 📜 Citation & Academic Disclosure

If you utilize this benchmark, dataset, or the `AgroStack-CR` algorithm in your research, please cite:

```bibtex
@misc{west_bengal_agri_ml_2026,
  author = {Final Year Computer Science Project Team},
  title  = {Integrating Soil Fertility, Climate and Geospatial Variables for Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal, India},
  year   = {2026},
  school = {Maulana Abul Kalam Azad University of Technology (MAKAUT), West Bengal}
}
```
