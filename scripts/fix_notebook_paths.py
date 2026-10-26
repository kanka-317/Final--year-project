"""
Fix path references across all Jupyter notebooks in notebooks/
Ensures seamless execution whether run from the project root or inside notebooks/ folder.
"""

import os
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
notebooks_dir = os.path.join(root_dir, 'notebooks')

# -------------------------------------------------------------
# 1. Fix soil_crop_fertilizer_complete_benchmark.ipynb
# -------------------------------------------------------------
nb_master_path = os.path.join(notebooks_dir, 'soil_crop_fertilizer_complete_benchmark.ipynb')
print(f"Patching {nb_master_path}...")
with open(nb_master_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0 (imports): Add working directory auto-fix
source_cell_0 = nb['cells'][1]['source'] # The first code cell
auto_dir_fix = [
    "# Ensure working directory is project root so all relative paths resolve seamlessly\n",
    "import os, sys\n",
    "if os.path.basename(os.getcwd()) == 'notebooks':\n",
    "    os.chdir('..')\n",
    "print(f\"Project Working Directory: {os.getcwd()}\")\n",
    "\n"
]
# Prepend if not already present
if not any("os.chdir('..')" in line for line in source_cell_0):
    nb['cells'][1]['source'] = auto_dir_fix + source_cell_0

# Cell 2 (dataset ingestion): Add robust multi-path candidate resolver
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "dataset_path = 'west_bengal_agriculture_master_20000_synthetic.csv'" in source_str:
            new_source = [
                "# Robust multi-path dataset resolver (supports running from root or notebooks/)\n",
                "candidates = [\n",
                "    'benchmark_data/west_bengal_agriculture_master_20000_synthetic.csv',\n",
                "    'west_bengal_agriculture_master_20000_synthetic.csv',\n",
                "    '../benchmark_data/west_bengal_agriculture_master_20000_synthetic.csv',\n",
                "    '../west_bengal_agriculture_master_20000_synthetic.csv',\n",
                "    'cleaned_data.csv',\n",
                "    '../cleaned_data.csv'\n",
                "]\n",
                "dataset_path = None\n",
                "for p in candidates:\n",
                "    if os.path.exists(p):\n",
                "        dataset_path = p\n",
                "        break\n",
                "\n",
                "if dataset_path is None:\n",
                "    raise FileNotFoundError('Could not locate west_bengal_agriculture_master_20000_synthetic.csv.')\n",
                "\n",
                "print(f\"Loading dataset from: {dataset_path}\")\n",
                "df_raw = pd.read_csv(dataset_path)\n",
                "\n",
                "print(f\"Loaded raw dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.\")\n",
                "display(df_raw.head(3))\n"
            ]
            cell['source'] = new_source
            print("  Updated dataset loading cell in master benchmark notebook.")
            break

with open(nb_master_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

# -------------------------------------------------------------
# 2. Fix part1_error_analysis_benchmark.ipynb
# -------------------------------------------------------------
nb_p1_path = os.path.join(notebooks_dir, 'part1_error_analysis_benchmark.ipynb')
print(f"Patching {nb_p1_path}...")
with open(nb_p1_path, 'r', encoding='utf-8') as f:
    nb1 = json.load(f)

# Cell 0: Prepend working directory normalization
source_cell_1_0 = nb1['cells'][1]['source']
if not any("os.chdir('..')" in line for line in source_cell_1_0):
    nb1['cells'][1]['source'] = auto_dir_fix + source_cell_1_0

for cell in nb1['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "df_raw = pd.read_csv('cleaned_data.csv')" in source_str:
            new_source = [
                "# Robust multi-path dataset resolver\n",
                "candidates = ['cleaned_data.csv', '../cleaned_data.csv', 'benchmark_data/west_bengal_agriculture_master_20000_synthetic.csv', '../benchmark_data/west_bengal_agriculture_master_20000_synthetic.csv']\n",
                "data_path = next((p for p in candidates if os.path.exists(p)), 'cleaned_data.csv')\n",
                "print(f\"Loading dataset from: {data_path}\")\n",
                "df_raw = pd.read_csv(data_path)\n",
                "print(f\"Loaded Master Dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns\")\n",
                "\n",
                "# Domain Feature Engineering Function\n",
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
                "    norm_pH = 1.0 - np.clip(np.abs(data['soil_pH'] - 6.8) / 3.0, 0, 1)\n",
                "    norm_OC = np.clip(data['soil_organic_carbon_pct'] / 1.0, 0, 1)\n",
                "    norm_N = np.clip(data['available_N_kg_ha'] / 400.0, 0, 1)\n",
                "    norm_P = np.clip(data['available_P_kg_ha'] / 50.0, 0, 1)\n",
                "    norm_K = np.clip(data['available_K_kg_ha'] / 250.0, 0, 1)\n",
                "    norm_micro = 1.0 - (data['total_micronutrient_deficiencies'] / 5.0)\n",
                "    data['calculated_SFI_score'] = (0.20 * norm_OC + 0.15 * norm_pH + 0.20 * norm_N + 0.15 * norm_P + 0.15 * norm_K + 0.15 * norm_micro) * 100.0\n",
                "    return data\n"
            ]
            cell['source'] = new_source
            print("  Updated dataset loading cell in part1 notebook.")
            break

with open(nb_p1_path, 'w', encoding='utf-8') as f:
    json.dump(nb1, f, indent=2)

# -------------------------------------------------------------
# 3. Fix part2_dataset_assessment_recommendations.ipynb
# -------------------------------------------------------------
nb_p2_path = os.path.join(notebooks_dir, 'part2_dataset_assessment_recommendations.ipynb')
print(f"Patching {nb_p2_path}...")
with open(nb_p2_path, 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

source_cell_2_0 = nb2['cells'][1]['source']
if not any("os.chdir('..')" in line for line in source_cell_2_0):
    nb2['cells'][1]['source'] = auto_dir_fix + source_cell_2_0

with open(nb_p2_path, 'w', encoding='utf-8') as f:
    json.dump(nb2, f, indent=2)

# -------------------------------------------------------------
# 4. Fix part3_proposed_algorithm_agrostack_cr.ipynb
# -------------------------------------------------------------
nb_p3_path = os.path.join(notebooks_dir, 'part3_proposed_algorithm_agrostack_cr.ipynb')
print(f"Patching {nb_p3_path}...")
with open(nb_p3_path, 'r', encoding='utf-8') as f:
    nb3 = json.load(f)

source_cell_3_0 = nb3['cells'][1]['source']
if not any("os.chdir('..')" in line for line in source_cell_3_0):
    nb3['cells'][1]['source'] = auto_dir_fix + source_cell_3_0

with open(nb_p3_path, 'w', encoding='utf-8') as f:
    json.dump(nb3, f, indent=2)

print("\nAll 4 notebooks successfully patched for seamless multi-path resolution!")
