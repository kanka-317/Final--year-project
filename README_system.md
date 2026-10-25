# Production Input Prediction Point System (`README_system.md`)

**Project Title:** Machine Learning Benchmark for Soil Fertility, Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal  
**Analytical Scope:** Standalone Production Inference Endpoints & Web Advisory Application  
**Component Reference:** Agent 8 Deliverables (`predict.py`, `app.py`, `tests/`)

---

## 1. System Overview

The prediction point engine provides a leak-free production interface for generating precision agricultural advisories across the 23 administrative districts of West Bengal. It embeds the fitted scikit-learn preprocessing pipeline (`fitted_pipeline.joblib`) and the top-performing ensemble model (`best_crop_model.joblib`), ensuring identical data transformations to those validated during empirical benchmarking.

### Core Advisory Capabilities
1. **Soil Fertility Index (SFI) Scoring:** Computes composite fertility index (0–100 scale) and classifies soil as *Low*, *Moderate*, or *High Fertility*.
2. **12-Parameter Nutrient Status Evaluation:** Benchmarks each nutrient against ICAR deficiency thresholds (e.g., $Zn < 0.60\text{ mg/kg}$, $B < 0.50\text{ mg/kg}$).
3. **Probabilistic Crop Suitability Ranking:** Returns Top-3 suitable crops with calibrated prediction confidence percentages.
4. **Primary Limiting Nutrient Identification:** Detects the most growth-limiting nutrient via Liebig’s Law of the Minimum.
5. **Site-Specific N-P-K Fertilizer Prescription:** Computes crop nutritional demand minus soil nutrient supply, outputting urea, SSP/DAP, and MOP equivalent dosages in $kg/ha$.
6. **Out-of-Distribution (OOD) Safeguards:** Flags inputs outside empirical soil boundaries ($pH < 3.8$ or $> 9.0$, rainfall $< 200\text{ mm}$), directing the user to official extension wet-lab testing.

---

## 2. Supported Inference Modes

### Mode 1: Manual Parameter Entry via CLI
Query single field measurements via command line:
```bash
python predict.py --mode manual
```

### Mode 2: Geospatial Coordinate Lookup
Query nearest soil profile and meteorological records using latitude and longitude:
```bash
python predict.py --mode geo --lat 23.24 --lon 87.86
```

### Mode 3: Batch Processing via CSV Upload
Execute high-throughput batch predictions across multiple soil survey samples:
```bash
python predict.py --mode batch --input_csv sample_field_tests.csv --output_csv batch_advisories.csv
```

### Mode 4: Interactive Streamlit Web Interface
Launch the browser-based graphical user interface:
```bash
streamlit run app.py
```

---

## 3. Unit Testing & Verification

Comprehensive unit tests validate both standard inputs and boundary edge cases:
```bash
python -m unittest tests/test_prediction.py
```
- Status: **5 Tests Passed (100% OK)**
- Verified coverage: Standard prediction, output validity, extreme acidic OOD warning, negative pH exception handling, and geospatial nearest-neighbor lookup.

---

## 4. Legal & Advisory Disclaimer

> **ADVISORY NOTICE:**  
> This software is an academic decision-support prototype. Recommendations should be cross-verified against official district Soil Health Card wet-chemistry reports and Block Agriculture Development Officer (BADO) agricultural guidelines prior to field capital investment.
