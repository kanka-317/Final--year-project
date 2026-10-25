# Data Dictionary: West Bengal Soil-Crop-Fertilizer Benchmark Dataset

**File Reference:** `west_bengal_agriculture_master_20000_synthetic.csv` / `cleaned_data.csv`  
**Dataset Description:** Master tabular dataset covering 20,000 observations across all 23 districts of West Bengal, India. Contains the 12 core soil physicochemical parameters, meso-climatic and geospatial attributes, crop classification targets, and site-specific fertilizer recommendations.

---

## 1. Soil Physicochemical Parameters (The Core 12 Attributes)

| # | Column Name | Symbol / Parameter | Standard Measurement Unit | Normal / Reference Range in WB Soils | Role in Machine Learning Pipeline |
|---|---|---|---|---|---|
| 1 | `soil_pH` | pH | Dimensionless (1:2.5 $H_2O$) | 4.5 – 7.5 (Acidic to Neutral) | Core Soil Input / SFI component |
| 2 | `soil_EC_dS_m` | Electrical Conductivity | dS/m (at 25°C) | 0.1 – 2.0 (< 1.0 normal, > 2.0 saline) | Core Soil Input / Salinity indicator |
| 3 | `soil_organic_carbon_pct` | Organic Carbon (OC) | % (Walkley-Black oxidizable) | 0.20 – 1.20% (Low <0.5, Medium 0.5-0.75, High >0.75) | Core Soil Input / Primary SFI driver |
| 4 | `available_N_kg_ha` | Available Nitrogen (N) | kg/ha ($KMnO_4$ oxidizable) | 100 – 400 (Low <280, Medium 280-560, High >560) | Macro-nutrient Input / Stoichiometric ratios |
| 5 | `available_P_kg_ha` | Available Phosphorus (P) | kg/ha (Bray P1 / Olsen) | 5 – 60 (Low <10, Medium 10-24.6, High >24.6) | Macro-nutrient Input / Limiting factor |
| 6 | `available_K_kg_ha` | Available Potassium (K) | kg/ha ($1N\\ NH_4OAc$ extractable) | 50 – 300 (Low <108, Medium 108-280, High >280) | Macro-nutrient Input / Osmotic regulator |
| 7 | `available_S_mg_kg` | Available Sulphur (S) | mg/kg or ppm ($0.15%\\ CaCl_2$) | 5 – 40 (Deficient <10, Adequate >10) | Secondary Nutrient Input |
| 8 | `Zn_mg_kg` | Available Zinc (Zn) | mg/kg or ppm (DTPA-extractable) | 0.2 – 2.5 (Critical limit: 0.60 mg/kg) | Micro-nutrient Input / Enzyme activator |
| 9 | `Fe_mg_kg` | Available Iron (Fe) | mg/kg or ppm (DTPA-extractable) | 5.0 – 50.0 (Critical limit: 4.50 mg/kg) | Micro-nutrient Input / Chlorophyll synthesis |
| 10 | `Mn_mg_kg` | Available Manganese (Mn) | mg/kg or ppm (DTPA-extractable) | 2.0 – 25.0 (Critical limit: 2.00 mg/kg) | Micro-nutrient Input / Photosystem driver |
| 11 | `Cu_mg_kg` | Available Copper (Cu) | mg/kg or ppm (DTPA-extractable) | 0.2 – 3.0 (Critical limit: 0.20 mg/kg) | Micro-nutrient Input / Redox catalyst |
| 12 | `B_mg_kg` | Available Boron (B) | mg/kg or ppm (Hot water soluble) | 0.1 – 1.8 (Critical limit: 0.50 mg/kg) | Micro-nutrient Input / Reproductive growth |

---

## 2. Geospatial, Climatic, and Environmental Attributes

| Column Name | Description | Units | Range / Categories |
|---|---|---|---|
| `latitude_approx` | Approximate sample latitude coordinate | Decimal Degrees | 21.80° N to 27.20° N |
| `longitude_approx` | Approximate sample longitude coordinate | Decimal Degrees | 86.25° E to 89.65° E |
| `district` | Administrative District of West Bengal | String (Categorical) | 23 Districts |
| `soil_zone` | Major Agro-Ecological / Pedological Zone | String (Categorical) | Alluvial, Terai, Lateritic, Coastal, Mountain |
| `season` | Cropping season | String (Categorical) | Kharif (Monsoon), Rabi (Winter), Mixed |
| `annual_mean_temperature_C` | Mean annual ambient temperature | °C | 7.6°C to 32.5°C |
| `annual_rainfall_mm` | Cumulative annual precipitation | mm | 31.5 mm to 4984.1 mm |
| `relative_humidity_pct` | Mean annual relative humidity | % | 35.0% to 98.0% |

---

## 3. Targets and Production Outcome Columns

| Column Name | Category | Role in Pipeline | Description |
|---|---|---|---|
| `crop` | Classification Target | **Supervised Target** | 12 classes: Rice, Wheat, Potato, Jute, Maize, Mustard, Vegetables, Tea, Lentil, Groundnut, Sugarcane, Sesame |
| `yield_kg_ha` | Continuous Metric | **Regression Target** | Harvested crop yield per hectare (kg/ha) |
| `production_tonnes` | Administrative Metric | Excluded (Leakage) | Total gross production in metric tonnes |
| `area_ha` | Administrative Metric | Excluded (Leakage) | Cultivated land area in hectares |
| `illustrative_SFI_0_100` | Continuous Index | Benchmark / Target | Baseline synthetic Soil Fertility Index score (0 - 100) |
| `recommended_N_kg_ha_simulated` | Continuous Metric | Multi-target Regressor | Simulated site-specific Nitrogen requirement (kg/ha) |
| `recommended_P_kg_ha_simulated` | Continuous Metric | Multi-target Regressor | Simulated site-specific Phosphorus requirement (kg/ha) |
| `recommended_K_kg_ha_simulated` | Continuous Metric | Multi-target Regressor | Simulated site-specific Potassium requirement (kg/ha) |
| `data_type` | Metadata | Drop | Labelled: `SYNTHETIC_SIMULATED` |
| `data_source` | Metadata | Drop | Model prototyping declaration |
| `research_use_warning` | Metadata | Drop | Advisory non-field experimental notice |
