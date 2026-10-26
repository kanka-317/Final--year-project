"""
Script to fix part1_error_analysis_benchmark.ipynb:
Ensures Cell 2 completely defines numeric_features, categorical_cols,
X_df, X_train_raw, X_test_raw, X_train, X_test, y_train, y_test, dist_train, n_classes,
so Cell 3 and all subsequent cells execute with zero NameErrors.
"""

import os
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
nb_path = os.path.join(root_dir, 'notebooks', 'part1_error_analysis_benchmark.ipynb')

print(f"Reading {nb_path}...")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The complete, robust Cell 2 source code
cell_2_source = [
    "# Ensure root working directory so all relative paths resolve cleanly\n",
    "import os, sys\n",
    "if os.path.basename(os.getcwd()) == 'notebooks':\n",
    "    os.chdir('..')\n",
    "\n",
    "# 1. Load Dataset\n",
    "candidates = [\n",
    "    'cleaned_data.csv',\n",
    "    'notebooks/cleaned_data.csv',\n",
    "    '../cleaned_data.csv',\n",
    "    'benchmark_data/west_bengal_agriculture_master_20000_synthetic.csv',\n",
    "    '../benchmark_data/west_bengal_agriculture_master_20000_synthetic.csv'\n",
    "]\n",
    "data_path = next((p for p in candidates if os.path.exists(p)), 'cleaned_data.csv')\n",
    "print(f\"Loading dataset from: {data_path}\")\n",
    "df_raw = pd.read_csv(data_path)\n",
    "print(f\"Loaded Master Dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns\")\n",
    "\n",
    "# 2. Domain Feature Engineering Function\n",
    "def engineer_agronomic_features(df):\n",
    "    data = df.copy()\n",
    "    eps = 1e-6\n",
    "    data['ratio_N_P'] = data['available_N_kg_ha'] / (data['available_P_kg_ha'] + eps)\n",
    "    data['ratio_N_K'] = data['available_N_kg_ha'] / (data['available_K_kg_ha'] + eps)\n",
    "    data['ratio_P_K'] = data['available_P_kg_ha'] / (data['available_K_kg_ha'] + eps)\n",
    "    data['ratio_OC_N'] = (data['soil_organic_carbon_pct'] * 1000) / (data['available_N_kg_ha'] + eps)\n",
    "    data['pH_neutral_deviation'] = np.abs(data['soil_pH'] - 7.0)\n",
    "    data['deficiency_Zn'] = (data['Zn_mg_kg'] < 0.60).astype(int)\n",
    "    data['deficiency_B'] = (data['B_mg_kg'] < 0.50).astype(int)\n",
    "    data['deficiency_Fe'] = (data['Fe_mg_kg'] < 4.50).astype(int)\n",
    "    data['deficiency_Cu'] = (data['Cu_mg_kg'] < 0.20).astype(int)\n",
    "    data['deficiency_Mn'] = (data['Mn_mg_kg'] < 2.00).astype(int)\n",
    "    data['total_micronutrient_deficiencies'] = (\n",
    "        data['deficiency_Zn'] + data['deficiency_B'] + data['deficiency_Fe'] +\n",
    "        data['deficiency_Cu'] + data['deficiency_Mn']\n",
    "    )\n",
    "    data['hydrothermal_ratio'] = data['annual_rainfall_mm'] / (data['annual_mean_temperature_C'] + 1.0)\n",
    "    data['vapor_pressure_proxy'] = (data['annual_mean_temperature_C'] * data['relative_humidity_pct']) / 100.0\n",
    "    \n",
    "    norm_pH = 1.0 - np.clip(np.abs(data['soil_pH'] - 6.8) / 3.0, 0, 1)\n",
    "    norm_OC = np.clip(data['soil_organic_carbon_pct'] / 1.0, 0, 1)\n",
    "    norm_N = np.clip(data['available_N_kg_ha'] / 400.0, 0, 1)\n",
    "    norm_P = np.clip(data['available_P_kg_ha'] / 50.0, 0, 1)\n",
    "    norm_K = np.clip(data['available_K_kg_ha'] / 250.0, 0, 1)\n",
    "    norm_micro = 1.0 - (data['total_micronutrient_deficiencies'] / 5.0)\n",
    "    \n",
    "    data['calculated_SFI_score'] = (\n",
    "        0.20 * norm_OC + 0.15 * norm_pH + 0.20 * norm_N +\n",
    "        0.15 * norm_P + 0.15 * norm_K + 0.15 * norm_micro\n",
    "    ) * 100.0\n",
    "    return data\n",
    "\n",
    "print(\"Engineering agronomic features...\")\n",
    "df_eng = engineer_agronomic_features(df_raw)\n",
    "\n",
    "# 3. Feature Definitions & Column Grouping\n",
    "soil_12_cols = [\n",
    "    'soil_pH', 'soil_EC_dS_m', 'soil_organic_carbon_pct',\n",
    "    'available_N_kg_ha', 'available_P_kg_ha', 'available_K_kg_ha', 'available_S_mg_kg',\n",
    "    'Zn_mg_kg', 'Fe_mg_kg', 'Mn_mg_kg', 'Cu_mg_kg', 'B_mg_kg'\n",
    "]\n",
    "climate_geo_cols = [\n",
    "    'latitude_approx', 'longitude_approx', 'annual_mean_temperature_C',\n",
    "    'annual_rainfall_mm', 'relative_humidity_pct'\n",
    "]\n",
    "engineered_cols = [\n",
    "    'ratio_N_P', 'ratio_N_K', 'ratio_P_K', 'ratio_OC_N',\n",
    "    'pH_neutral_deviation', 'deficiency_Zn', 'deficiency_B', 'deficiency_Fe',\n",
    "    'deficiency_Cu', 'deficiency_Mn', 'total_micronutrient_deficiencies',\n",
    "    'hydrothermal_ratio', 'vapor_pressure_proxy', 'calculated_SFI_score'\n",
    "]\n",
    "categorical_cols = ['season', 'soil_zone']\n",
    "numeric_features = soil_12_cols + climate_geo_cols + engineered_cols\n",
    "all_features_raw = numeric_features + categorical_cols\n",
    "\n",
    "X_df = df_eng[all_features_raw].copy()\n",
    "y_crop_raw = df_eng['crop'].copy()\n",
    "districts = df_eng['district'].copy()\n",
    "\n",
    "label_enc = LabelEncoder()\n",
    "y_encoded = label_enc.fit_transform(y_crop_raw)\n",
    "class_names = list(label_enc.classes_)\n",
    "n_classes = len(class_names)\n",
    "\n",
    "# 4. 70/30 Stratified Train/Test Split\n",
    "GLOBAL_SEED = 42\n",
    "X_train_raw, X_test_raw, y_train, y_test, dist_train, dist_test = train_test_split(\n",
    "    X_df, y_encoded, districts, test_size=0.30, stratify=y_encoded, random_state=GLOBAL_SEED\n",
    ")\n",
    "\n",
    "# 5. Clean Preprocessor (Fitted strictly on Training Split)\n",
    "preprocessor_clean = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', Pipeline([\n",
    "            ('scaler', RobustScaler()),\n",
    "            ('power', PowerTransformer(method='yeo-johnson'))\n",
    "        ]), numeric_features),\n",
    "        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)\n",
    "    ]\n",
    ")\n",
    "X_train = preprocessor_clean.fit_transform(X_train_raw)\n",
    "X_test = preprocessor_clean.transform(X_test_raw)\n",
    "\n",
    "print(f\"Preprocessed Clean Features: X_train={X_train.shape}, X_test={X_test.shape}\")\n",
    "print(f\"Defined: numeric_features ({len(numeric_features)}), categorical_cols ({len(categorical_cols)}), n_classes={n_classes}\")\n"
]

# Update cell 2 in part1 notebook
found = False
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "def engineer_agronomic_features(df):" in source_str:
            cell['source'] = cell_2_source
            found = True
            print(f"Updated Cell #{idx} with complete numeric_features and train/test split definitions.")
            break

# Also check precomputed JSON load paths in later cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "pd.read_json('deliverables/classifier_20_errors.json')" in line:
                new_source.append("clf_path = 'deliverables/classifier_20_errors.json' if os.path.exists('deliverables/classifier_20_errors.json') else '../deliverables/classifier_20_errors.json'\n")
                new_source.append("df_errors_clf = pd.read_json(clf_path)\n")
            elif "pd.read_json('deliverables/regressor_20_errors.json')" in line:
                new_source.append("reg_path = 'deliverables/regressor_20_errors.json' if os.path.exists('deliverables/regressor_20_errors.json') else '../deliverables/regressor_20_errors.json'\n")
                new_source.append("df_errors_reg = pd.read_json(reg_path)\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("part1_error_analysis_benchmark.ipynb successfully patched and saved!")
