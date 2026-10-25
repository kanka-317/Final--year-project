"""
Clean and organize workspace files into distinct, logical directories without duplicates.
Moves files out of root into clean subdirectories.
"""

import os
import shutil

root_dir = os.path.dirname(os.path.abspath(__file__))

# Define destination directories and the files that belong in them
mapping = {
    'notebooks': [
        'part1_error_analysis_benchmark.ipynb',
        'part2_dataset_assessment_recommendations.ipynb',
        'soil_crop_fertilizer_complete_benchmark.ipynb'
    ],
    'manuscripts_and_reports': [
        'manuscript.docx',
        'cover_letter.docx',
        'supplementary.docx',
        'comprehensive_error_analysis_and_benchmark_report.docx',
        'algorithm_limitations_comprehensive_report.docx',
        'model_training_report.docx'
    ],
    'scripts': [
        'create_master_notebook.py',
        'create_separate_notebooks.py',
        'build_full_docx_report.py',
        'build_algorithm_limitations_docx.py',
        'build_manuscript_and_submission_docs.py',
        'generate_docx_report.py',
        'compute_full_benchmark_and_errors.py',
        'run_complete_error_analysis.py',
        'make_all_figures.py',
        'run_all.py',
        'organize_project_structure.py'
    ],
    'docs': [
        'algorithm_assessment.md',
        'bias_variance_report.md',
        'leakage_audit.md',
        'cleaning_log.md',
        'preprocessing_report.md',
        'dataset_description.md',
        'data_dictionary.md',
        'feature_report.md',
        'correlation_report.md',
        'best_algorithm_recommendation.md',
        'gap_statement.md',
        'error_report.md',
        'figure_summary.md',
        'figure_index.md',
        'submission_checklist.md',
        'mock_review.md',
        'response_to_reviewers.md',
        'references.bib',
        'research_paper_summaries.json'
    ],
    'benchmark_data': [
        'west_bengal_agriculture_master_20000_synthetic.csv',
        'supervised_metrics.csv',
        'unsupervised_metrics.csv',
        'rl_metrics.csv',
        'best_params.json',
        'error_analysis_12_parameters.xlsx',
        'error_master_table.xlsx',
        'literature_table.xlsx'
    ]
}

# Remove any obsolete nested folders created previously
old_folders = [os.path.join(root_dir, 'docs', 'benchmark_analysis'), os.path.join(root_dir, 'scripts', 'benchmark_generation')]
for of in old_folders:
    if os.path.exists(of):
        for f in os.listdir(of):
            src = os.path.join(of, f)
            parent = os.path.dirname(of)
            dst = os.path.join(parent, f)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(of, ignore_errors=True)

print("Moving files to proper folders (NO duplicates)...")
for folder, files in mapping.items():
    dest_dir = os.path.join(root_dir, folder)
    os.makedirs(dest_dir, exist_ok=True)
    for fname in files:
        src = os.path.join(root_dir, fname)
        dst = os.path.join(dest_dir, fname)
        if os.path.exists(src):
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.move(src, dst)
                print(f"  Moved: {fname} -> {folder}/")

print("\nWorkspace clean and organized!")
