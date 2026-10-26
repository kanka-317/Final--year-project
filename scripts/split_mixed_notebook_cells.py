import json
import glob
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
notebooks = glob.glob(os.path.join(root_dir, 'notebooks', '*.ipynb'))

for nb_path in sorted(notebooks):
    nb_name = os.path.basename(nb_path)
    print(f"\nProcessing {nb_name}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    new_cells = []
    modified = False
    
    for idx, cell in enumerate(nb['cells']):
        source_text = ''.join(cell.get('source', []))
        if 'if os.path.basename(os.getcwd()) == \'notebooks\':' in source_text and '##' in source_text:
            lines = cell['source']
            code_lines = []
            md_lines = []
            is_code = True
            for line in lines:
                if line.startswith('##'):
                    is_code = False
                if is_code:
                    code_lines.append(line)
                else:
                    md_lines.append(line)
            
            # Create pure code cell
            cell_code = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_lines
            }
            # Create pure markdown cell
            cell_md = {
                "cell_type": "markdown",
                "metadata": {},
                "source": md_lines
            }
            new_cells.append(cell_code)
            new_cells.append(cell_md)
            modified = True
            print(f"  Cleanly split mixed cell {idx} into Code Cell + Markdown Cell.")
        else:
            new_cells.append(cell)
            
    if modified:
        nb['cells'] = new_cells
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print(f"  Successfully updated {nb_name}")
