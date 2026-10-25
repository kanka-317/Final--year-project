import os
import sys
import time
import shutil
import subprocess

def run_step(step_name, command):
    print(f"\n=======================================================")
    print(f"STEP: {step_name}")
    print(f"COMMAND: {command}")
    print(f"=======================================================")
    t0 = time.time()
    res = subprocess.run(command, shell=True)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"ERROR: {step_name} failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"SUCCESS: {step_name} completed in {elapsed:.2f} seconds.")

def main():
    print("*******************************************************************")
    print("WEST BENGAL SOIL-CROP-FERTILIZER BENCHMARK: ONE-COMMAND REPRODUCER")
    print("Academic Affiliation: MAKAUT CSE Final-Year Research Project")
    print("*******************************************************************")
    t_start = time.time()

    os.makedirs('deliverables', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('figures/svg', exist_ok=True)

    # 1. Run Complete Error Analysis & Leakage Audit
    run_step("Empirical Error Analysis & Leakage Audit", "python compute_full_benchmark_and_errors.py")

    # 2. Generate Complete Figure Suite & Heatmaps
    run_step("Generate Scientific Heatmaps H1-H16 & Plots", "python make_all_figures.py")

    # 3. Generate Standalone Jupyter Notebooks
    run_step("Build Standalone Jupyter Notebooks (.ipynb)", "python create_separate_notebooks.py")

    # 4. Generate Comprehensive Research Paper DOCX Report
    run_step("Build Comprehensive DOCX Benchmark Report", "python build_full_docx_report.py")

    # 5. Generate Journal Manuscript, Supplementary & Cover Letter
    run_step("Build Journal Manuscript, Supplementary & Cover Letter (.docx)", "python build_manuscript_and_submission_docs.py")

    # 6. Execute Unit Tests on Prediction Engine
    run_step("Run Unit Tests on Prediction Engine", "python -m unittest tests/test_prediction.py")

    # 7. Synchronize all deliverables to /deliverables
    print("\nSynchronizing final deliverables into /deliverables directory...")
    files_to_sync = [
        "cleaned_data.csv", "cleaning_log.md", "data_dictionary.md",
        "literature_table.xlsx", "references.bib", "gap_statement.md",
        "preprocessing_report.md", "feature_report.md", "error_report.md",
        "supervised_metrics.csv", "unsupervised_metrics.csv", "rl_metrics.csv",
        "dataset_description.md", "algorithm_assessment.md", "best_algorithm_recommendation.md",
        "error_master_table.xlsx", "bias_variance_report.md", "leakage_audit.md",
        "error_analysis_12_parameters.xlsx", "best_params.json",
        "figure_index.md", "figure_summary.md", "correlation_report.md",
        "manuscript.docx", "supplementary.docx", "cover_letter.docx",
        "mock_review.md", "response_to_reviewers.md", "submission_checklist.md",
        "README_system.md", "predict.py", "app.py",
        "part1_error_analysis_benchmark.ipynb", "part2_dataset_assessment_recommendations.ipynb"
    ]
    for fn in files_to_sync:
        if os.path.exists(fn):
            shutil.copy2(fn, os.path.join("deliverables", fn))
            
    total_elapsed = time.time() - t_start
    print("\n*******************************************************************")
    print(f"PIPELINE COMPLETE: All tables, figures, models & reports reproduced in {total_elapsed:.2f}s!")
    print(f"Total synchronized deliverables in /deliverables: {len(os.listdir('deliverables'))}")
    print("*******************************************************************")

if __name__ == '__main__':
    main()
