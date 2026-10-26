import json
import glob
import re
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("Root directory:", root_dir)

notebooks = glob.glob(os.path.join(root_dir, 'notebooks', '*.ipynb'))

for nb_file in sorted(notebooks):
    print(f"\nScanning {os.path.basename(nb_file)}...")
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            lines = cell['source']
            new_lines = []
            for line in lines:
                # Fix error_analysis_12_parameters.xlsx path
                if 'error_analysis_12_parameters.xlsx' in line:
                    line = re.sub(
                        r"['\"][^'\"]*error_analysis_12_parameters\.xlsx['\"]",
                        "next(p for p in ['benchmark_data/error_analysis_12_parameters.xlsx', '../benchmark_data/error_analysis_12_parameters.xlsx', 'deliverables/error_analysis_12_parameters.xlsx'] if os.path.exists(p))",
                        line
                    )
                    modified = True
                # Fix error_master_table.xlsx path
                if 'error_master_table.xlsx' in line:
                    line = re.sub(
                        r"['\"][^'\"]*error_master_table\.xlsx['\"]",
                        "next(p for p in ['benchmark_data/error_master_table.xlsx', '../benchmark_data/error_master_table.xlsx', 'deliverables/error_master_table.xlsx'] if os.path.exists(p))",
                        line
                    )
                    modified = True
                # Fix literature_table.xlsx path
                if 'literature_table.xlsx' in line:
                    line = re.sub(
                        r"['\"][^'\"]*literature_table\.xlsx['\"]",
                        "next(p for p in ['benchmark_data/literature_table.xlsx', '../benchmark_data/literature_table.xlsx'] if os.path.exists(p))",
                        line
                    )
                    modified = True
                new_lines.append(line)
            cell['source'] = new_lines
            
    if modified:
        with open(nb_file, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print(f"Patched paths in {os.path.basename(nb_file)} successfully.")
    else:
        print(f"No changes needed in {os.path.basename(nb_file)}.")
