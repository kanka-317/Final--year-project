import json
import glob
import os
import re

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
notebooks = glob.glob(os.path.join(root_dir, 'notebooks', '*.ipynb'))

for nb_path in sorted(notebooks):
    nb_name = os.path.basename(nb_path)
    print(f"\nProcessing {nb_name}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    # 1. Convert any markdown cell containing python code for working directory into a code cell
    for idx, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'markdown':
            s = ''.join(cell.get('source', []))
            if 'os.chdir(' in s and 'os.path.basename(os.getcwd())' in s:
                cell['cell_type'] = 'code'
                cell['execution_count'] = None
                cell['outputs'] = []
                cell['metadata'] = {}
                print(f"  Converted Cell {idx} from markdown to code cell.")
                modified = True
    
    # 2. In part2_dataset_assessment_recommendations.ipynb, make all file loads robust
    if nb_name == 'part2_dataset_assessment_recommendations.ipynb':
        for idx, cell in enumerate(nb['cells']):
            if cell.get('cell_type') == 'code':
                s = ''.join(cell.get('source', []))
                # First code cell (Dataset Profiling Statistics)
                if 'dataset_description_stats.json' in s:
                    cell['source'] = [
                        "import os, sys, json\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "if os.path.basename(os.getcwd()) == 'notebooks':\n",
                        "    os.chdir('..')\n",
                        "\n",
                        "# Load Dataset Profiling Statistics with path resolver\n",
                        "p_stat = next(p for p in ['deliverables/dataset_description_stats.json', '../deliverables/dataset_description_stats.json'] if os.path.exists(p))\n",
                        "with open(p_stat, 'r') as f:\n",
                        "    data_card = json.load(f)\n",
                        "\n",
                        "print(f\"Dataset Dimensions: {data_card['shape']['rows']} rows x {data_card['shape']['columns']} columns\")\n",
                        "print(f\"Memory Footprint: {data_card['memory_mb']:.2f} MB\")\n",
                        "print(f\"Target Class Imbalance Ratio: {data_card['crop_distribution']['imbalance_ratio']:.3f} (Balanced across 12 crops)\")\n",
                        "\n",
                        "# Column-by-Column Profiling Table\n",
                        "df_cols = pd.DataFrame(data_card['columns'])\n",
                        "display(df_cols[['name', 'type', 'missing_pct', 'mean', 'median', 'std', 'min', 'max', 'skew', 'outlier_count']].head(15))\n"
                    ]
                    modified = True
                    print(f"  Injected path resolver and chdir into Cell {idx} (dataset_description_stats.json).")
                
                # Classifier & Regressor errors
                elif 'classifier_20_errors.json' in s:
                    lines = cell.get('source', [])
                    new_lines = []
                    for line in lines:
                        if "pd.read_json('deliverables/classifier_20_errors.json')" in line:
                            new_lines.append("p_clf = next(p for p in ['deliverables/classifier_20_errors.json', '../deliverables/classifier_20_errors.json'] if os.path.exists(p))\n")
                            new_lines.append("df_errors_clf = pd.read_json(p_clf)\n")
                            modified = True
                        elif "pd.read_json('deliverables/regressor_20_errors.json')" in line:
                            new_lines.append("p_reg = next(p for p in ['deliverables/regressor_20_errors.json', '../deliverables/regressor_20_errors.json'] if os.path.exists(p))\n")
                            new_lines.append("df_errors_reg = pd.read_json(p_reg)\n")
                            modified = True
                        else:
                            new_lines.append(line)
                    cell['source'] = new_lines
                    print(f"  Injected path resolver into Cell {idx} (classifier/regressor errors).")
                
                # Statistical tests
                elif 'statistical_tests.json' in s:
                    lines = cell.get('source', [])
                    new_lines = []
                    for line in lines:
                        if "open('deliverables/statistical_tests.json'" in line:
                            new_lines.append("p_stat = next(p for p in ['deliverables/statistical_tests.json', '../deliverables/statistical_tests.json'] if os.path.exists(p))\n")
                            new_lines.append("with open(p_stat, 'r') as f:\n")
                            modified = True
                        else:
                            new_lines.append(line)
                    cell['source'] = new_lines
                    print(f"  Injected path resolver into Cell {idx} (statistical_tests.json).")

    # 3. Ensure all notebooks write back cleanly
    if modified:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print(f"Saved {nb_name} with all fixes.")
    else:
        print(f"No changes required for {nb_name}.")
