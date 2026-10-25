"""
Interactive Quick-Test for Proposed Algorithm (AgroStack-CR)
Allows instant testing of any soil & climate sample to inspect:
1. Bayesian Minimum Expected Cost (MEC) Crop Recommendation (Ratio 4:1)
2. Calibrated Prediction Probabilities
3. 90% Conformal Safe Prediction Set (Co-Viable Portfolio)
4. Agronomic Uncertainty Rejection Status
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from proposed_algorithm import AgroStackCR

print("=" * 70)
print("TESTING PROPOSED ALGORITHM: AgroStack-CR")
print("=" * 70)

# 1. Load Preprocessing Pipeline
pipeline_path = os.path.join(root_dir, 'deliverables', 'fitted_pipeline.joblib')
if not os.path.exists(pipeline_path):
    pipeline_path = os.path.join(root_dir, 'fitted_pipeline.joblib')

pipeline = joblib.load(pipeline_path)
print(f"Loaded Preprocessing Pipeline: {pipeline_path}")

# 2. Load Master Dataset and Fit AgroStack-CR (Fast demo mode or full)
data_path = os.path.join(root_dir, 'cleaned_data.csv')
print(f"Loading dataset from: {data_path}")
df_raw = pd.read_csv(data_path)

# Domain feature engineering
def engineer_features(df):
    data = df.copy()
    eps = 1e-6
    data['ratio_N_P'] = data['available_N_kg_ha'] / (data['available_P_kg_ha'] + eps)
    data['ratio_N_K'] = data['available_N_kg_ha'] / (data['available_K_kg_ha'] + eps)
    data['ratio_P_K'] = data['available_P_kg_ha'] / (data['available_K_kg_ha'] + eps)
    data['ratio_OC_N'] = (data['soil_organic_carbon_pct'] * 1000) / (data['available_N_kg_ha'] + eps)
    data['pH_neutral_deviation'] = np.abs(data['soil_pH'] - 7.0)
    data['deficiency_Zn'] = (data['Zn_mg_kg'] < 0.60).astype(int)
    data['deficiency_B'] = (data['B_mg_kg'] < 0.50).astype(int)
    data['deficiency_Fe'] = (data['Fe_mg_kg'] < 4.50).astype(int)
    data['deficiency_Cu'] = (data['Cu_mg_kg'] < 0.20).astype(int)
    data['deficiency_Mn'] = (data['Mn_mg_kg'] < 2.00).astype(int)
    data['total_micronutrient_deficiencies'] = (
        data['deficiency_Zn'] + data['deficiency_B'] + data['deficiency_Fe'] +
        data['deficiency_Cu'] + data['deficiency_Mn']
    )
    data['hydrothermal_ratio'] = data['annual_rainfall_mm'] / (data['annual_mean_temperature_C'] + 1.0)
    data['vapor_pressure_proxy'] = (data['annual_mean_temperature_C'] * data['relative_humidity_pct']) / 100.0
    
    norm_pH = 1.0 - np.clip(np.abs(data['soil_pH'] - 6.8) / 3.0, 0, 1)
    norm_OC = np.clip(data['soil_organic_carbon_pct'] / 1.0, 0, 1)
    norm_N = np.clip(data['available_N_kg_ha'] / 400.0, 0, 1)
    norm_P = np.clip(data['available_P_kg_ha'] / 50.0, 0, 1)
    norm_K = np.clip(data['available_K_kg_ha'] / 250.0, 0, 1)
    norm_micro = 1.0 - (data['total_micronutrient_deficiencies'] / 5.0)
    
    data['calculated_SFI_score'] = (
        0.20 * norm_OC + 0.15 * norm_pH + 0.20 * norm_N +
        0.15 * norm_P + 0.15 * norm_K + 0.15 * norm_micro
    ) * 100.0
    return data

df_eng = engineer_features(df_raw)

soil_12_cols = ['soil_pH', 'soil_EC_dS_m', 'soil_organic_carbon_pct', 'available_N_kg_ha', 'available_P_kg_ha', 'available_K_kg_ha', 'available_S_mg_kg', 'Zn_mg_kg', 'Fe_mg_kg', 'Mn_mg_kg', 'Cu_mg_kg', 'B_mg_kg']
climate_geo_cols = ['latitude_approx', 'longitude_approx', 'annual_mean_temperature_C', 'annual_rainfall_mm', 'relative_humidity_pct']
engineered_cols = ['ratio_N_P', 'ratio_N_K', 'ratio_P_K', 'ratio_OC_N', 'pH_neutral_deviation', 'deficiency_Zn', 'deficiency_B', 'deficiency_Fe', 'deficiency_Cu', 'deficiency_Mn', 'total_micronutrient_deficiencies', 'hydrothermal_ratio', 'vapor_pressure_proxy', 'calculated_SFI_score']
categorical_cols = ['season', 'soil_zone']

X_df = df_eng[soil_12_cols + climate_geo_cols + engineered_cols + categorical_cols]
y_crop = df_eng['crop'].values

# Preprocess features
X_trans = pipeline.transform(X_df)

# Train AgroStack-CR (Fast sample for demonstration)
print("\nFitting AgroStack-CR on 3,000 training samples...")
sub_indices = np.random.RandomState(42).choice(len(X_trans), 3000, replace=False)
model = AgroStackCR(n_estimators_et=40, max_iter_hgb=40, C_FP=4.0, alpha=0.10, random_state=42)
model.fit(X_trans[sub_indices], y_crop[sub_indices])

# 3. Test on Sample Field Queries
print("\n" + "=" * 70)
print("TEST RESULTS ON SAMPLE SOIL QUERIES:")
print("=" * 70)

test_indices = [10, 42, 105, 500]
for idx in test_indices:
    sample_vec = X_trans[idx:idx+1]
    true_crop = y_crop[idx]
    
    # Run predictions
    point_pred, conf_set, status = model.predict_with_status(sample_vec)
    probs = model.predict_proba(sample_vec)[0]
    top_3_idx = np.argsort(probs)[::-1][:3]
    top_3_str = ", ".join([f"{model.classes_[i]}: {probs[i]*100:.1f}%" for i in top_3_idx])
    
    print(f"\n[Field Sample #{idx}]")
    print(f"  Ground Truth Crop         : {true_crop}")
    print(f"  AgroStack-CR Point Pick   : {point_pred[0]} (Cost-Optimal under 4:1 Risk)")
    print(f"  Top-3 Calibrated Probs    : {top_3_str}")
    print(f"  90% Conformal Safe Set    : {list(conf_set[0])} ({len(conf_set[0])} viable crops)")
    print(f"  Advisory Status           : {status[0]}")
    print(f"  Ground Truth in Safe Set? : {'YES (Safe Portfolio)' if true_crop in conf_set[0] else 'NO'}")

print("\n" + "=" * 70)
print("TEST COMPLETED SUCCESSFULLY!")
print("=" * 70)
