# Data Cleaning Log: West Bengal Soil-Crop-Fertilizer Dataset

**Dataset Processed:** `west_bengal_agriculture_master_20000_synthetic.csv`  
**Output Dataset:** `cleaned_data.csv`  
**Initial Dimensions:** 20,000 rows × 34 columns  
**Final Cleaned Dimensions:** 20,000 rows × 34 columns (with identified administrative and leakage partitions)  
**Execution Timestamp:** 2026-10-23  

---

## 1. Inspection and Summary Statistics
- **Row count:** 20,000 records.
- **Column count:** 34 columns.
- **Missing / NaN values:** 0 null values detected across all 34 attributes.
- **Duplicate Records:** 0 exact duplicate rows across all features.
- **Unique Identifier:** `record_id` (SYNWB00001 through SYNWB20000).

---

## 2. Standardisation of Column Names, Units, and District Names
- All column headers follow consistent snake_case format with embedded units:
  - `soil_pH` (dimensionless, 1:2.5 soil-water suspension)
  - `soil_EC_dS_m` (Decisiemens per meter at 25°C)
  - `soil_organic_carbon_pct` (Walkley-Black oxidizable organic carbon, %)
  - `available_N_kg_ha`, `available_P_kg_ha`, `available_K_kg_ha` (Alkaline permanganate N, Bray/Olsen P, Ammonium acetate K in kg/ha)
  - `available_S_mg_kg`, `Zn_mg_kg`, `Fe_mg_kg`, `Mn_mg_kg`, `Cu_mg_kg`, `B_mg_kg` (DTPA-extractable and hot water soluble in mg/kg / ppm)
  - `annual_mean_temperature_C` (°C)
  - `annual_rainfall_mm` (mm)
  - `relative_humidity_pct` (%)
  - `yield_kg_ha` (kg/ha)
- District spellings were audited against Census of India and West Bengal administrative gazettes across all 23 districts (e.g., Alipurduar, Bankura, Birbhum, Cooch Behar, Dakshin Dinajpur, Darjeeling, Hooghly, Howrah, Jalpaiguri, Jhargram, Kalimpong, Kolkata, Malda, Murshidabad, Nadia, North 24 Parganas, Paschim Bardhaman, Paschim Medinipur, Purba Bardhaman, Purba Medinipur, Purulia, South 24 Parganas, Uttar Dinajpur).

---

## 3. Domain Range Boundary Audit (Impossible Values Check)

| Parameter | Domain Valid Range | Dataset Min | Dataset Max | Violations (< Min) | Violations (> Max) | Action Taken | Rationale |
|---|---|---|---|---|---|---|---|
| `soil_pH` | [0.0, 14.0] | 4.00 | 8.50 | 0 | 0 | Retained | Fully within valid agronomic pH range (acidic Terai to neutral/alkaline alluvial) |
| `soil_EC_dS_m` | [0.0, 15.0] | 0.064 | 3.00 | 0 | 0 | Retained | Non-saline to moderately saline coastal soils |
| `soil_organic_carbon_pct` | [0.0, 5.0] | 0.100 | 1.477 | 0 | 0 | Retained | Realistic for tropical/subtropical agro-ecosystems |
| `available_N_kg_ha` | [0.0, 1000.0] | 50.00 | 436.40 | 0 | 0 | Retained | Low to medium availability, standard for West Bengal |
| `available_P_kg_ha` | [0.0, 300.0] | 3.00 | 80.09 | 0 | 0 | Retained | Realistic soil test phosphorus levels |
| `available_K_kg_ha` | [0.0, 1000.0] | 20.00 | 337.55 | 0 | 0 | Retained | Covers deficient laterites to fertile riverine alluvium |
| `available_S_mg_kg` | [0.0, 200.0] | 2.00 | 52.01 | 0 | 0 | Retained | Typical available sulphate sulfur concentrations |
| `Zn_mg_kg` | [0.0, 50.0] | 0.100 | 2.350 | 0 | 0 | Retained | DTPA-extractable zinc spans deficient (<0.6) to adequate |
| `Fe_mg_kg` | [0.0, 200.0] | 2.00 | 45.62 | 0 | 0 | Retained | Ample available iron typical of acidic alluvial/lateritic soils |
| `Mn_mg_kg` | [0.0, 100.0] | 1.00 | 20.08 | 0 | 0 | Retained | DTPA-extractable manganese |
| `Cu_mg_kg` | [0.0, 50.0] | 0.100 | 2.580 | 0 | 0 | Retained | Copper levels within natural soil thresholds |
| `B_mg_kg` | [0.0, 20.0] | 0.050 | 1.510 | 0 | 0 | Retained | Hot water extractable boron |
| `annual_mean_temperature_C` | [-5.0, 50.0] | 7.66 | 32.49 | 0 | 0 | Retained | Captures hilly sub-Himalayan zones to plains |
| `annual_rainfall_mm` | [0.0, 8000.0] | 31.47 | 4984.08 | 0 | 0 | Retained | Covers monsoon precipitation from Rarh to Dooars |
| `relative_humidity_pct` | [0.0, 100.0] | 35.00 | 98.00 | 0 | 0 | Retained | Plausible tropical humidity ranges |

---

## 4. Target Leakage and Feature Partitioning Strategy
To adhere strictly to **Rule 5 (No Data Leakage)**:
- **Primary Classification Target:** `crop` (12 balanced classes: Vegetables, Rice, Tea, Jute, Maize, Mustard, Potato, Lentil, Groundnut, Sugarcane, Wheat, Sesame).
- **Leakage Columns Excluded from Features:**
  - `record_id`: Arbitrary primary key.
  - `state`: Uniform constant ('West Bengal').
  - `district`: Administrative boundary; dropped to evaluate generalizeable agro-edaphic features (can be used for spatial hold-out CV).
  - `yield_kg_ha`, `production_tonnes`, `area_ha`: Post-harvest production outcomes (predicting crop from yield would represent severe target leakage).
  - `illustrative_SFI_0_100`: Synthetic proxy SFI (isolated as a regression target or rule benchmark).
  - `recommended_N_kg_ha_simulated`, `recommended_P_kg_ha_simulated`, `recommended_K_kg_ha_simulated`: Simulated fertilizer prescriptions (isolated as multi-target regression outcomes).
  - `data_type`, `data_source`, `research_use_warning`: Synthetic metadata tags.
