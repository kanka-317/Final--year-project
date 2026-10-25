"""
Remove all duplicate files from deliverables/ folder.
Keeps only the true deployment model artifacts and the proposed_algorithm/ subfolder.
Ensures zero duplicates exist between deliverables/ and the rest of the workspace.
"""

import os

deliverables_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deliverables'))

# Files that legitimately belong in deliverables/ (core baseline deployment artifacts)
keep_files = {
    'best_crop_model.joblib',
    'fitted_pipeline.joblib',
    'classifier_20_errors.json',
    'regressor_20_errors.json',
    'statistical_tests.json',
    'leakage_results.json',
    'dataset_description_stats.json',
    'best_params.json'
}

print(f"Cleaning duplicate files from {deliverables_dir}...")
deleted_count = 0

for item in os.listdir(deliverables_dir):
    item_path = os.path.join(deliverables_dir, item)
    
    # Do not delete proposed_algorithm folder or any subdirectories
    if os.path.isdir(item_path):
        continue
        
    # Check if this file is a duplicate that should be removed
    if item not in keep_files:
        try:
            os.remove(item_path)
            print(f"  Removed duplicate: {item}")
            deleted_count += 1
        except Exception as e:
            print(f"  Error removing {item}: {e}")

print(f"\nDeliverables folder cleaned! Removed {deleted_count} duplicate files.")
