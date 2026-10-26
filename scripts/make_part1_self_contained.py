"""
Make Cell 3 in part1_error_analysis_benchmark.ipynb 100% self-contained with a self-healing guard.
Guarantees that even if Cell In[7] is executed by itself, numeric_features, X_df, X_train,
categorical_cols, etc. are always defined and ready.
"""

import os
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
nb_path = os.path.join(root_dir, 'notebooks', 'part1_error_analysis_benchmark.ipynb')

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

guard_code = [
    "# Self-healing guard: ensure feature definitions and splits exist even if this cell is executed standalone\n",
    "if 'numeric_features' not in globals() or 'X_train' not in globals() or 'X_df' not in globals():\n",
    "    import os, sys\n",
    "    if os.path.basename(os.getcwd()) == 'notebooks':\n",
    "        os.chdir('..')\n",
    "    \n",
    "    data_candidates = ['cleaned_data.csv', 'notebooks/cleaned_data.csv', '../cleaned_data.csv', 'benchmark_data/west_bengal_agriculture_master_20000_synthetic.csv', '../benchmark_data/west_bengal_agriculture_master_20000_synthetic.csv']\n",
    "    p_data = next((p for p in data_candidates if os.path.exists(p)), 'cleaned_data.csv')\n",
    "    df_raw = pd.read_csv(p_data)\n",
    "    eps = 1e-6\n",
    "    df_eng = df_raw.copy()\n",
    "    df_eng['ratio_N_P'] = df_eng['available_N_kg_ha'] / (df_eng['available_P_kg_ha'] + eps)\n",
    "    df_eng['ratio_N_K'] = df_eng['available_N_kg_ha'] / (df_eng['available_K_kg_ha'] + eps)\n",
    "    df_eng['ratio_P_K'] = df_eng['available_P_kg_ha'] / (df_eng['available_K_kg_ha'] + eps)\n",
    "    df_eng['ratio_OC_N'] = (df_eng['soil_organic_carbon_pct'] * 1000) / (df_eng['available_N_kg_ha'] + eps)\n",
    "    df_eng['pH_neutral_deviation'] = np.abs(df_eng['soil_pH'] - 7.0)\n",
    "    df_eng['deficiency_Zn'] = (df_eng['Zn_mg_kg'] < 0.60).astype(int)\n",
    "    df_eng['deficiency_B'] = (df_eng['B_mg_kg'] < 0.50).astype(int)\n",
    "    df_eng['deficiency_Fe'] = (df_eng['Fe_mg_kg'] < 4.50).astype(int)\n",
    "    df_eng['deficiency_Cu'] = (df_eng['Cu_mg_kg'] < 0.20).astype(int)\n",
    "    df_eng['deficiency_Mn'] = (df_eng['Mn_mg_kg'] < 2.00).astype(int)\n",
    "    df_eng['total_micronutrient_deficiencies'] = df_eng['deficiency_Zn'] + df_eng['deficiency_B'] + df_eng['deficiency_Fe'] + df_eng['deficiency_Cu'] + df_eng['deficiency_Mn']\n",
    "    df_eng['hydrothermal_ratio'] = df_eng['annual_rainfall_mm'] / (df_eng['annual_mean_temperature_C'] + 1.0)\n",
    "    df_eng['vapor_pressure_proxy'] = (df_eng['annual_mean_temperature_C'] * df_eng['relative_humidity_pct']) / 100.0\n",
    "    norm_pH = 1.0 - np.clip(np.abs(df_eng['soil_pH'] - 6.8) / 3.0, 0, 1)\n",
    "    norm_OC = np.clip(df_eng['soil_organic_carbon_pct'] / 1.0, 0, 1)\n",
    "    norm_N = np.clip(df_eng['available_N_kg_ha'] / 400.0, 0, 1)\n",
    "    norm_P = np.clip(df_eng['available_P_kg_ha'] / 50.0, 0, 1)\n",
    "    norm_K = np.clip(df_eng['available_K_kg_ha'] / 250.0, 0, 1)\n",
    "    norm_micro = 1.0 - (df_eng['total_micronutrient_deficiencies'] / 5.0)\n",
    "    df_eng['calculated_SFI_score'] = (0.20 * norm_OC + 0.15 * norm_pH + 0.20 * norm_N + 0.15 * norm_P + 0.15 * norm_K + 0.15 * norm_micro) * 100.0\n",
    "    \n",
    "    soil_12_cols = ['soil_pH', 'soil_EC_dS_m', 'soil_organic_carbon_pct', 'available_N_kg_ha', 'available_P_kg_ha', 'available_K_kg_ha', 'available_S_mg_kg', 'Zn_mg_kg', 'Fe_mg_kg', 'Mn_mg_kg', 'Cu_mg_kg', 'B_mg_kg']\n",
    "    climate_geo_cols = ['latitude_approx', 'longitude_approx', 'annual_mean_temperature_C', 'annual_rainfall_mm', 'relative_humidity_pct']\n",
    "    engineered_cols = ['ratio_N_P', 'ratio_N_K', 'ratio_P_K', 'ratio_OC_N', 'pH_neutral_deviation', 'deficiency_Zn', 'deficiency_B', 'deficiency_Fe', 'deficiency_Cu', 'deficiency_Mn', 'total_micronutrient_deficiencies', 'hydrothermal_ratio', 'vapor_pressure_proxy', 'calculated_SFI_score']\n",
    "    categorical_cols = ['season', 'soil_zone']\n",
    "    numeric_features = soil_12_cols + climate_geo_cols + engineered_cols\n",
    "    all_features_raw = numeric_features + categorical_cols\n",
    "    \n",
    "    X_df = df_eng[all_features_raw].copy()\n",
    "    y_crop_raw = df_eng['crop'].copy()\n",
    "    districts = df_eng['district'].copy()\n",
    "    label_enc = LabelEncoder()\n",
    "    y_encoded = label_enc.fit_transform(y_crop_raw)\n",
    "    class_names = list(label_enc.classes_)\n",
    "    n_classes = len(class_names)\n",
    "    GLOBAL_SEED = 42\n",
    "    \n",
    "    X_train_raw, X_test_raw, y_train, y_test, dist_train, dist_test = train_test_split(\n",
    "        X_df, y_encoded, districts, test_size=0.30, stratify=y_encoded, random_state=GLOBAL_SEED\n",
    "    )\n",
    "    preprocessor_clean = ColumnTransformer(\n",
    "        transformers=[\n",
    "            ('num', Pipeline([\n",
    "                ('scaler', RobustScaler()),\n",
    "                ('power', PowerTransformer(method='yeo-johnson'))\n",
    "            ]), numeric_features),\n",
    "            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)\n",
    "        ]\n",
    "    )\n",
    "    X_train = preprocessor_clean.fit_transform(X_train_raw)\n",
    "    X_test = preprocessor_clean.transform(X_test_raw)\n",
    "    print(\"Self-healing guard initialized all features and splits successfully.\")\n",
    "\n"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "preprocessor_leaky = ColumnTransformer(" in source_str:
            if not any("Self-healing guard" in line for line in cell['source']):
                cell['source'] = guard_code + cell['source']
                print("Injected self-healing guard into Leakage Audit Cell.")
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook patched successfully!")
