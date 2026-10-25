import os
import sys
import argparse
import joblib
import json
import numpy as np
import pandas as pd

MODEL_PATH = 'deliverables/best_crop_model.joblib'
PIPELINE_PATH = 'deliverables/fitted_pipeline.joblib'
DATASET_PATH = 'cleaned_data.csv'

eps = 1e-6

# FAO / ICAR Standard Crop Nutrient Demands (kg/ha)
CROP_FERTILIZER_STANDARDS = {
    'Rice': {'req_N': 120.0, 'req_P': 50.0, 'req_K': 45.0},
    'Wheat': {'req_N': 130.0, 'req_P': 60.0, 'req_K': 50.0},
    'Potato': {'req_N': 180.0, 'req_P': 80.0, 'req_K': 120.0},
    'Jute': {'req_N': 100.0, 'req_P': 40.0, 'req_K': 40.0},
    'Maize': {'req_N': 140.0, 'req_P': 60.0, 'req_K': 60.0},
    'Mustard': {'req_N': 90.0, 'req_P': 40.0, 'req_K': 30.0},
    'Vegetables': {'req_N': 150.0, 'req_P': 70.0, 'req_K': 90.0},
    'Tea': {'req_N': 140.0, 'req_P': 45.0, 'req_K': 60.0},
    'Lentil': {'req_N': 25.0, 'req_P': 50.0, 'req_K': 20.0},
    'Groundnut': {'req_N': 30.0, 'req_P': 60.0, 'req_K': 40.0},
    'Sugarcane': {'req_N': 220.0, 'req_P': 90.0, 'req_K': 120.0},
    'Sesame': {'req_N': 40.0, 'req_P': 30.0, 'req_K': 20.0}
}

CROP_CLASSES = [
    'Groundnut', 'Jute', 'Lentil', 'Maize', 'Mustard', 'Potato',
    'Rice', 'Sesame', 'Sugarcane', 'Tea', 'Vegetables', 'Wheat'
]

def load_system():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError("Model or Pipeline artifact not found. Please verify deliverables folder.")
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline

def engineer_features_single(df):
    data = df.copy()
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

def evaluate_nutrient_status(soil_dict):
    status = {}
    # N
    n = soil_dict['available_N_kg_ha']
    status['Nitrogen'] = 'Low' if n < 280 else ('Medium' if n <= 560 else 'High')
    # P
    p = soil_dict['available_P_kg_ha']
    status['Phosphorus'] = 'Low' if p < 10 else ('Medium' if p <= 25 else 'High')
    # K
    k = soil_dict['available_K_kg_ha']
    status['Potassium'] = 'Low' if k < 108 else ('Medium' if k <= 280 else 'High')
    # OC
    oc = soil_dict['soil_organic_carbon_pct']
    status['Organic_Carbon'] = 'Low' if oc < 0.50 else ('Medium' if oc <= 0.75 else 'High')
    # Micronutrients
    status['Zinc'] = 'Deficient' if soil_dict['Zn_mg_kg'] < 0.60 else 'Sufficient'
    status['Boron'] = 'Deficient' if soil_dict['B_mg_kg'] < 0.50 else 'Sufficient'
    status['Iron'] = 'Deficient' if soil_dict['Fe_mg_kg'] < 4.50 else 'Sufficient'
    return status

def predict_point(soil_dict, season='Kharif', soil_zone='Alluvial'):
    model, pipeline = load_system()
    
    # Range validations
    if soil_dict['soil_pH'] < 0 or soil_dict['soil_pH'] > 14:
        raise ValueError("Invalid soil pH. Must be within [0.0, 14.0].")
    
    row_df = pd.DataFrame([soil_dict])
    row_df['season'] = season
    row_df['soil_zone'] = soil_zone
    
    # Feature engineering
    row_eng = engineer_features_single(row_df)
    
    # Preprocessing
    soil_12_cols = ['soil_pH', 'soil_EC_dS_m', 'soil_organic_carbon_pct', 'available_N_kg_ha', 'available_P_kg_ha', 'available_K_kg_ha', 'available_S_mg_kg', 'Zn_mg_kg', 'Fe_mg_kg', 'Mn_mg_kg', 'Cu_mg_kg', 'B_mg_kg']
    climate_geo_cols = ['latitude_approx', 'longitude_approx', 'annual_mean_temperature_C', 'annual_rainfall_mm', 'relative_humidity_pct']
    engineered_cols = ['ratio_N_P', 'ratio_N_K', 'ratio_P_K', 'ratio_OC_N', 'pH_neutral_deviation', 'deficiency_Zn', 'deficiency_B', 'deficiency_Fe', 'deficiency_Cu', 'deficiency_Mn', 'total_micronutrient_deficiencies', 'hydrothermal_ratio', 'vapor_pressure_proxy', 'calculated_SFI_score']
    numeric_features = soil_12_cols + climate_geo_cols + engineered_cols
    categorical_features = ['season', 'soil_zone']
    
    X_point = pipeline.transform(row_eng[numeric_features + categorical_features])
    
    # Probability prediction
    probs = model.predict_proba(X_point)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(CROP_CLASSES[i], round(float(probs[i]) * 100, 2)) for i in top3_idx]
    
    # SFI
    sfi = float(row_eng['calculated_SFI_score'].values[0])
    sfi_class = 'High Fertility' if sfi > 70 else ('Moderate Fertility' if sfi > 45 else 'Low Fertility')
    
    # Limiting nutrient
    nutrients = {
        'Nitrogen': soil_dict['available_N_kg_ha'] / 280.0,
        'Phosphorus': soil_dict['available_P_kg_ha'] / 23.0,
        'Potassium': soil_dict['available_K_kg_ha'] / 140.0,
        'Zinc': soil_dict['Zn_mg_kg'] / 0.60,
        'Boron': soil_dict['B_mg_kg'] / 0.50
    }
    limiting_nutrient = min(nutrients, key=nutrients.get)
    
    # Fertilizer prescription for Top-1 Crop (Demand - Soil Supply)
    top_crop = top3[0][0]
    std_req = CROP_FERTILIZER_STANDARDS.get(top_crop, {'req_N': 100.0, 'req_P': 40.0, 'req_K': 40.0})
    fert_N = max(0.0, std_req['req_N'] - soil_dict['available_N_kg_ha'] * 0.30)
    fert_P = max(0.0, std_req['req_P'] - soil_dict['available_P_kg_ha'] * 0.20)
    fert_K = max(0.0, std_req['req_K'] - soil_dict['available_K_kg_ha'] * 0.15)
    
    # Out of Distribution Warning
    ood = bool(soil_dict['soil_pH'] < 3.8 or soil_dict['soil_pH'] > 9.0 or soil_dict['annual_rainfall_mm'] < 200.0 or soil_dict['annual_rainfall_mm'] > 5500.0)
    
    return {
        'Soil_Fertility_Index': round(sfi, 2),
        'Fertility_Classification': sfi_class,
        'Nutrient_Status': evaluate_nutrient_status(soil_dict),
        'Top_3_Suitable_Crops': [{'Crop': c, 'Confidence_Pct': p} for c, p in top3],
        'Primary_Limiting_Nutrient': limiting_nutrient,
        'Recommended_Fertilizer_kg_ha': {
            'N (Urea basis)': round(fert_N, 2),
            'P2O5 (SSP/DAP basis)': round(fert_P, 2),
            'K2O (MOP basis)': round(fert_K, 2)
        },
        'OOD_Uncertainty_Warning': ood,
        'Advisory_Disclaimer': 'Model-based decision support. Confirm prescriptions with official Block Agriculture Development Officer (BADO) guidance.'
    }

def predict_by_coordinates(lat, lon, season='Kharif'):
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Database {DATASET_PATH} not found.")
    df_db = pd.read_csv(DATASET_PATH)
    dists = np.sqrt((df_db['latitude_approx'] - lat)**2 + (df_db['longitude_approx'] - lon)**2)
    nearest_idx = dists.idxmin()
    nearest_row = df_db.loc[nearest_idx]
    
    soil_dict = {
        'soil_pH': float(nearest_row['soil_pH']),
        'soil_EC_dS_m': float(nearest_row['soil_EC_dS_m']),
        'soil_organic_carbon_pct': float(nearest_row['soil_organic_carbon_pct']),
        'available_N_kg_ha': float(nearest_row['available_N_kg_ha']),
        'available_P_kg_ha': float(nearest_row['available_P_kg_ha']),
        'available_K_kg_ha': float(nearest_row['available_K_kg_ha']),
        'available_S_mg_kg': float(nearest_row['available_S_mg_kg']),
        'Zn_mg_kg': float(nearest_row['Zn_mg_kg']),
        'Fe_mg_kg': float(nearest_row['Fe_mg_kg']),
        'Mn_mg_kg': float(nearest_row['Mn_mg_kg']),
        'Cu_mg_kg': float(nearest_row['Cu_mg_kg']),
        'B_mg_kg': float(nearest_row['B_mg_kg']),
        'latitude_approx': lat,
        'longitude_approx': lon,
        'annual_mean_temperature_C': float(nearest_row['annual_mean_temperature_C']),
        'annual_rainfall_mm': float(nearest_row['annual_rainfall_mm']),
        'relative_humidity_pct': float(nearest_row['relative_humidity_pct'])
    }
    res = predict_point(soil_dict, season=season, soil_zone=nearest_row['soil_zone'])
    res['Nearest_District'] = nearest_row['district']
    res['Nearest_Soil_Zone'] = nearest_row['soil_zone']
    res['Interpolation_Distance_Deg'] = round(float(dists.min()), 4)
    return res

def batch_predict_csv(input_csv_path, output_csv_path):
    df_in = pd.read_csv(input_csv_path)
    print(f"Loaded batch file {input_csv_path} with {len(df_in)} rows.")
    results = []
    for idx, row in df_in.iterrows():
        s_dict = row.to_dict()
        res = predict_point(s_dict, season=row.get('season', 'Kharif'), soil_zone=row.get('soil_zone', 'Alluvial'))
        results.append({
            'Index': idx,
            'Top_Crop': res['Top_3_Suitable_Crops'][0]['Crop'],
            'Confidence': res['Top_3_Suitable_Crops'][0]['Confidence_Pct'],
            'SFI_Score': res['Soil_Fertility_Index'],
            'Rec_N': res['Recommended_Fertilizer_kg_ha']['N (Urea basis)'],
            'Rec_P': res['Recommended_Fertilizer_kg_ha']['P2O5 (SSP/DAP basis)'],
            'Rec_K': res['Recommended_Fertilizer_kg_ha']['K2O (MOP basis)']
        })
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv_path, index=False)
    print(f"Batch prediction results saved to {output_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="West Bengal Soil-Crop-Fertilizer Prediction Engine")
    parser.add_argument('--mode', type=str, choices=['manual', 'geo', 'batch'], default='manual', help="Inference mode")
    parser.add_argument('--lat', type=float, default=23.24, help="Latitude coordinate")
    parser.add_argument('--lon', type=float, default=87.86, help="Longitude coordinate")
    parser.add_argument('--input_csv', type=str, help="Input CSV path for batch mode")
    parser.add_argument('--output_csv', type=str, default='batch_predictions.csv', help="Output CSV path")
    
    args = parser.parse_args()
    
    if args.mode == 'manual':
        sample_input = {
            'soil_pH': 5.85, 'soil_EC_dS_m': 0.42, 'soil_organic_carbon_pct': 0.65,
            'available_N_kg_ha': 190.0, 'available_P_kg_ha': 18.5, 'available_K_kg_ha': 110.0,
            'available_S_mg_kg': 14.2, 'Zn_mg_kg': 0.45, 'Fe_mg_kg': 16.0, 'Mn_mg_kg': 6.5,
            'Cu_mg_kg': 0.85, 'B_mg_kg': 0.38, 'latitude_approx': 23.24, 'longitude_approx': 87.86,
            'annual_mean_temperature_C': 25.4, 'annual_rainfall_mm': 1420.0, 'relative_humidity_pct': 72.0
        }
        res = predict_point(sample_input, season='Kharif', soil_zone='Alluvial')
        print("\n--- Manual Prediction Output ---")
        print(json.dumps(res, indent=4))
    elif args.mode == 'geo':
        res = predict_by_coordinates(args.lat, args.lon)
        print("\n--- Geospatial Nearest-Neighbor Prediction Output ---")
        print(json.dumps(res, indent=4))
    elif args.mode == 'batch':
        if not args.input_csv:
            print("Error: --input_csv required for batch mode.")
        else:
            batch_predict_csv(args.input_csv, args.output_csv)
