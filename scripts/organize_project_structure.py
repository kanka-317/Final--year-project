"""
Organize workspace files into clean, professional directory structure.
Copies/arranges files into logical subfolders while maintaining root references
to ensure full backwards compatibility with app.py and predict.py.
"""

import os
import shutil

root_dir = os.path.dirname(os.path.abspath(__file__))

structure = {
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
    'scripts/benchmark_generation': [
        'create_master_notebook.py',
        'create_separate_notebooks.py',
        'build_full_docx_report.py',
        'build_algorithm_limitations_docx.py',
        'build_manuscript_and_submission_docs.py',
        'generate_docx_report.py',
        'compute_full_benchmark_and_errors.py',
        'run_complete_error_analysis.py',
        'make_all_figures.py'
    ],
    'docs/benchmark_analysis': [
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
        'submission_checklist.md'
    ]
}

print("Organizing project folder structure...")
for folder, files in structure.items():
    folder_path = os.path.join(root_dir, folder)
    os.makedirs(folder_path, exist_ok=True)
    for fname in files:
        src = os.path.join(root_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(folder_path, fname)
            # Copy to preserve root compatibility
            shutil.copy2(src, dst)
            print(f"  -> Organised: {fname} -> {folder}/")

print("Project organization completed successfully!")
