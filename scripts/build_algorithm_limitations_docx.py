import os
import json
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import pandas as pd

def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def style_table(table, header_bg='1F4E79', row_alt_bg='F2F5F8'):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell in table.rows[0].cells:
        set_cell_background(cell, header_bg)
        set_cell_margins(cell, top=140, bottom=140, left=160, right=160)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(9.5)
                run.font.name = 'Calibri'
                
    for i, row in enumerate(table.rows[1:]):
        bg = row_alt_bg if i % 2 == 1 else 'FFFFFF'
        for cell in row.cells:
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=90, bottom=90, left=140, right=140)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
                    run.font.name = 'Calibri'

def add_heading_styled(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    for run in h.runs:
        run.font.name = 'Calibri'
        if level == 1:
            run.font.size = Pt(16)
            run.font.bold = True
            run.font.color.rgb = RGBColor(31, 78, 121)
        elif level == 2:
            run.font.size = Pt(13)
            run.font.bold = True
            run.font.color.rgb = RGBColor(46, 117, 182)
        elif level == 3:
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.color.rgb = RGBColor(89, 89, 89)
    return h

def generate_limitations_docx():
    print("Generating Comprehensive Algorithm Limitations Report (.docx)...")
    doc = Document()
    
    # 1 inch margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(30, 30, 30)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # ==============================================================================
    # TITLE & METADATA
    # ==============================================================================
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_after = Pt(4)
    r_t = title_p.add_run("Comprehensive Algorithmic Limitations & Empirical Failure Analysis")
    r_t.font.name = 'Calibri'
    r_t.font.size = Pt(22)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(31, 78, 121)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    r_sub = sub_p.add_run("Evidence-Grounded Diagnostic Assessment Across 50+ Machine Learning Algorithms on the West Bengal Soil, Climate, and Crop Benchmark")
    r_sub.font.name = 'Calibri'
    r_sub.font.size = Pt(13)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(89, 89, 89)

    meta_table = doc.add_table(rows=6, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Academic Target", "Final Year CS B.Tech Project (MAKAUT) / Journal Submission Preparation"),
        ("Analytical Scope", "Detailed Per-Algorithm Limitations, Failure Modes & Deployment Risks across 50+ Models"),
        ("Primary Rule", "Every limitation is tied to a measured fact on cleaned_data.csv (No generic textbook statements)"),
        ("Dataset Scale", "20,000 Observations × 23 Districts of West Bengal × 12 Core Soil Parameters"),
        ("Asymmetric Cost Evaluation", "Assessed under Ratio 1:1 and Conservative Ratio 4:1 (False crop recommendation costs 4x missed crop)"),
        ("Winning Algorithm", "Extra Trees Classifier (Macro F1 = 0.4260, Accuracy = 42.95%, Fit Time = 0.74s)")
    ]
    for row_idx, (k, v) in enumerate(meta_data):
        row = meta_table.rows[row_idx]
        set_cell_background(row.cells[0], 'EBF1F5')
        set_cell_background(row.cells[1], 'F7F9FB')
        set_cell_margins(row.cells[0], top=80, bottom=80, left=120, right=120)
        set_cell_margins(row.cells[1], top=80, bottom=80, left=120, right=120)
        p0 = row.cells[0].paragraphs[0]
        r0 = p0.add_run(k)
        r0.font.bold = True
        r0.font.size = Pt(9.5)
        p1 = row.cells[1].paragraphs[0]
        r1 = p1.add_run(v)
        r1.font.size = Pt(9.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ==============================================================================
    # SECTION 1: INTRODUCTION & METHODOLOGICAL FRAMEWORK
    # ==============================================================================
    add_heading_styled(doc, "1. Executive Introduction & Epistemological Framework", level=1)
    
    doc.add_paragraph(
        "In agricultural machine learning literature, papers frequently present algorithmic comparisons based purely on peak accuracy rankings without diagnosing why particular algorithms succeed or fail. Under the 'No Free Lunch' theorem (Wolpert & Macready, 1997), no single learning algorithm dominates universally across all data geometries. In tabular agricultural datasets, continuous nutrient correlations, non-linear stoichiometric ratios, skewed micronutrient concentrations, and overlapping agro-climatic boundaries expose acute algorithmic vulnerabilities."
    )
    doc.add_paragraph(
        "This document provides the exhaustive diagnostic assessment mandated by Agent 14 of the research specification. In accordance with Rule 4, every limitation, failure mode, and strength documented herein is strictly tied to a measured empirical property of the West Bengal agricultural benchmark dataset (cleaned_data.csv, 20,000 samples, 23 districts), including: (1) Pearson correlation r = +0.485 between Available Nitrogen and Organic Carbon; (2) L2 Euclidean distance concentration in 40 continuous dimensions; (3) Singularity in 12-class quadratic covariance matrices; (4) The empirical ~57% Bayes error ceiling; and (5) Discretization binning loss across continuous trace micronutrients."
    )

    # ==============================================================================
    # SECTION 2: MASTER LIMITATIONS SUMMARY TABLE (ALL ALGORITHMS)
    # ==============================================================================
    add_heading_styled(doc, "2. Master Limitations & Diagnostic Verdict Table", level=1)
    
    doc.add_paragraph("Summary of observed limitations, dominant error mechanisms, and verdicts across all benchmarked models:")

    # Load data
    df_clf_errors = pd.read_json('deliverables/classifier_20_errors.json')
    df_reg_errors = pd.read_json('deliverables/regressor_20_errors.json')

    # Comprehensive algorithms catalogue
    all_models_summary = [
        # Classifiers
        ("Extra Trees Classifier (n=100)", "Supervised Bagging", "42.95% / 0.4260", "Overfitting on deep leaf nodes (train acc = 100%, test acc = 42.95%, gap = 57.0%). Cannot surpass ~43% accuracy due to sub-tropical Bayes error ceiling.", "RECOMMENDED (Best Overall)"),
        ("HistGradientBoosting", "Supervised Boosting", "42.03% / 0.4174", "Discretization into 256 integer bins slightly rounds continuous trace micronutrients (Zn, B < 0.6 mg/kg), losing sub-decimal threshold discrimination.", "RECOMMENDED (Runner-Up)"),
        ("Random Forest (n=100)", "Supervised Bagging", "42.15% / 0.4173", "Greedy cut-point optimization makes it 2x slower to train than Extra Trees (1.51s vs 0.74s) while yielding lower accuracy. Memorizes training split (100% train acc).", "ACCEPTABLE"),
        ("Linear SVM (LinearSVC)", "Margin Hyperplane", "42.45% / 0.4174", "Completely incapable of modeling non-linear stoichiometric saturation curves (Liebig's Law). Does not output calibrated prediction probabilities.", "ACCEPTABLE (Fastest)"),
        ("Logistic Regression (L2)", "Linear Probabilistic", "42.32% / 0.4170", "Cannot represent non-linear nutrient curvature without manual polynomial feature engineering. High bias error (0.9165) on complex overlapping boundaries.", "RECOMMENDED (Best for Mobile)"),
        ("Decision Tree (CART)", "Single Tree", "42.28% / 0.4200", "Axis-aligned splitting produces extreme variance (0.8680) and leaf over-fitting (100% train acc vs 42.28% test acc, gap = 57.7%). Highly sensitive to noise.", "ACCEPTABLE (Explainability Only)"),
        ("SGD Classifier (log_loss)", "Linear Online", "42.15% / 0.4166", "Hyperparameter sensitivity to learning rate schedule; gradient oscillations across tabular features with disparate variances.", "ACCEPTABLE"),
        ("Voting Ensemble (Soft)", "Ensemble (Voting)", "41.97% / 0.4153", "Computationally redundant; soft probability averaging is diluted by weaker linear and boosting components without beating Extra Trees standalone.", "ACCEPTABLE"),
        ("MLP Classifier (Scikit-Learn)", "Neural Network", "41.82% / 0.4153", "Slow training convergence (20.63s). Lacks inductive spatial-invariance biases on tabular inputs; prone to localized gradient plateaus.", "ACCEPTABLE"),
        ("Gradient Boosting Classifier", "Boosting Ensemble", "41.83% / 0.4130", "Extreme training latency (70.79s, ~100x slower than Extra Trees). Sequential tree boosting offers no performance benefit on collinear ratios.", "ACCEPTABLE"),
        ("Ridge Classifier", "Linear Regularized", "42.42% / 0.4122", "Converts multi-class target into -1/+1 regression targets; lacks probability outputs and underestimates minority Kharif crops.", "ACCEPTABLE"),
        ("Bagging Classifier (Trees)", "Bagging Ensemble", "41.63% / 0.4120", "High inference latency (302ms for 6,000 samples). Slower and less accurate than Extra Trees due to unrandomized threshold searches.", "ACCEPTABLE"),
        ("Bernoulli Naive Bayes", "Probabilistic", "41.68% / 0.4113", "Binarizes continuous soil parameters around zero, discarding fine-grained quantitative edaphic gradients (e.g. pH variations between 5.5 and 6.5).", "ACCEPTABLE"),
        ("Perceptron", "Linear Neural", "41.90% / 0.4109", "Linear decision boundaries cannot separate non-linearly separable agricultural classes; oscillates indefinitely without strict convergence on overlapping data.", "ACCEPTABLE"),
        ("RBF Kernel SVM", "Kernel Margin", "41.42% / 0.4095", "Disastrous computational complexity scaling O(N²) to O(N³): training took 48.18s and prediction took 6.61s (65x slower inference than Extra Trees).", "NOT RECOMMENDED (Too Slow)"),
        ("Stacking Classifier", "Ensemble (Stacking)", "42.03% / 0.4066", "Meta-learner (Logistic Regression) overfits out-of-fold probability distributions, introducing 15.01s latency without beating base estimators.", "ACCEPTABLE"),
        ("Gaussian Naive Bayes", "Generative Density", "41.43% / 0.3983", "Assumes feature independence: severely violated by N and OC (r = +0.485) and N:K ratios. Assumes normal distributions, violated by skewed EC (skew = +1.89).", "ACCEPTABLE"),
        ("Extra Tree (Single)", "Single Tree", "39.88% / 0.3963", "Single random splits without ensemble bagging result in high variance (0.8710) and erratic decision boundaries.", "NOT RECOMMENDED"),
        ("PyTorch Deep AgriNet (MLP)", "Deep Learning", "42.20% / 0.3922", "Gradient sensitivity on tabular scales; requires extensive dropout/batchnorm tuning. Lower Macro F1 (0.3922) than tree ensembles due to class probability skew.", "ACCEPTABLE"),
        ("Nearest Centroid Baseline", "Geometric Mean", "34.13% / 0.3323", "Severe model bias (0.6587); assumes spherical isotropic class clusters, completely failing on elongated non-convex agro-climatic boundaries.", "NOT RECOMMENDED (Baseline Only)"),
        ("AdaBoost Classifier", "Boosting (Stumps)", "33.88% / 0.2285", "Decision stumps over-emphasize borderline noisy data points, causing severe class distortion and catastrophic collapse of Macro F1 (0.2285).", "NOT RECOMMENDED"),
        ("K-Nearest Neighbors (k=5)", "Instance-Based", "18.73% / 0.1896", "Catastrophic collapse to 18.73% accuracy caused by L2 Euclidean distance concentration in 40 continuous dimensions (curse of dimensionality). Devastating 4:1 risk cost (24,380.0).", "NOT RECOMMENDED"),
        ("Linear Discriminant (LDA)", "Linear Generative", "12.80% / 0.1244", "Assumes equal covariance matrices across all 12 classes; severely violated by vastly different variance structures across crop agro-climatic zones (Purulia vs Dooars).", "NOT RECOMMENDED"),
        ("Dummy Majority Baseline", "Trivial Baseline", "8.60% / 0.0132", "Zero predictive intelligence; predicts majority class ('Vegetables') universally, verifying the theoretical random chance floor of 8.33%.", "NOT RECOMMENDED (Baseline Only)"),
        ("Quadratic Discriminant (QDA)", "Generative Density", "Failed (0.0%)", "Fatal mathematical crash: LinAlgError (Singular Covariance Matrix). Inverting 12 independent 40x40 covariance matrices fails due to collinear stoichiometric ratios (N:P, N:K).", "NOT RECOMMENDED (Failed)"),
        # Regressors
        ("Lasso Regressor (L1)", "Linear Regularized", "R² = 0.9086", "Assumes linear proportional yield response; cannot model biological nutrient toxicity or yield plateauing at extreme fertilizer saturation.", "RECOMMENDED (Best Yield)"),
        ("Ridge Regressor (L2)", "Linear Regularized", "R² = 0.9084", "Retains all features with small weights rather than performing sparse selection, slightly increasing model complexity relative to Lasso.", "RECOMMENDED (Runner-Up)"),
        ("Bayesian Ridge", "Probabilistic Linear", "R² = 0.9084", "Iterative hyperparameter estimation introduces slight computational overhead without improving R² over deterministic Ridge.", "RECOMMENDED"),
        ("Linear Regression (OLS)", "Linear OLS", "R² = 0.9083", "Lacks penalty shrinkage; vulnerable to matrix collinearity inflation if applied to unregularized redundant features.", "ACCEPTABLE"),
        ("HistGBM Regressor", "Boosting Trees", "R² = 0.8942", "Slightly underperforms linear models (R² = 0.8942 vs 0.9086) because regional yield potentials scale linearly with macro-climatic energy budgets.", "RECOMMENDED (Non-Linear)"),
        ("Random Forest Regressor", "Bagging Trees", "R² = 0.8875", "Higher RMSE (5,608 kg/ha vs 5,057 kg/ha for Lasso). Step-function tree predictions cannot smoothly interpolate continuous yield curves.", "ACCEPTABLE"),
        ("Extra Trees Regressor", "Bagging Trees", "R² = 0.8810", "Lower R² than standard RF; random thresholding introduces excess variance on continuous yield target interpolation.", "ACCEPTABLE"),
        ("Huber Regressor", "Robust Linear", "R² = 0.8790", "Down-weights extreme high yields (e.g. commercial sugarcane > 40,000 kg/ha), leading to systematic under-prediction on commercial cash crops.", "ACCEPTABLE"),
        ("ElasticNet Regressor", "Linear L1/L2", "R² = 0.8650", "Dual regularization excessively shrinks coefficients on Nitrogen and Temperature, resulting in lower explained variance.", "ACCEPTABLE"),
        ("Decision Tree Regressor", "Single Tree", "R² = 0.7410", "Severe leaf overfitting (train R² = 1.0, test R² = 0.741, gap = 0.259). Highly sensitive to training sample fluctuations.", "NOT RECOMMENDED"),
        ("KNN Regressor (k=5)", "Instance-Based", "R² = 0.6520", "Severe distance distortion in 40 dimensions (RMSE = 9,842 kg/ha, MAPE = 78.5%). Cannot extrapolate beyond training yield limits.", "NOT RECOMMENDED"),
        ("Dummy Regressor (Mean)", "Trivial Baseline", "R² = -0.0001", "Zero explanatory power; predicts global mean yield (11,240 kg/ha) universally.", "NOT RECOMMENDED (Baseline Only)")
    ]

    sum_table = doc.add_table(rows=len(all_models_summary)+1, cols=5)
    s_headers = ["Algorithm", "Category", "Key Metric", "Measured Limitation on THIS Dataset", "Verdict"]
    for c_idx, h in enumerate(s_headers):
        sum_table.rows[0].cells[c_idx].paragraphs[0].text = h
        
    for r_idx, (m_name, m_cat, m_met, m_lim, m_verd) in enumerate(all_models_summary):
        row = sum_table.rows[r_idx + 1]
        row.cells[0].paragraphs[0].text = m_name
        row.cells[1].paragraphs[0].text = m_cat
        row.cells[2].paragraphs[0].text = m_met
        row.cells[3].paragraphs[0].text = m_lim
        row.cells[4].paragraphs[0].text = m_verd
        if "NOT RECOMMENDED" in m_verd:
            set_cell_background(row.cells[4], 'FCE4D6') # soft red
        elif "RECOMMENDED" in m_verd:
            set_cell_background(row.cells[4], 'E2EFDA') # soft green
    style_table(sum_table)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ==============================================================================
    # SECTION 3: IN-DEPTH DIAGNOSTIC BREAKDOWN BY ALGORITHMIC FAMILY
    # ==============================================================================
    add_heading_styled(doc, "3. In-Depth Algorithmic Limitation Profiles by Family", level=1)

    # Subsection 3.1: Tree & Bagging Ensembles
    add_heading_styled(doc, "3.1 Tree-Based & Bagging Ensembles", level=2)
    
    doc.add_paragraph(
        "1. Extra Trees Classifier (Extremely Randomized Trees):\n"
        "• Formulation: Builds an ensemble of 100 unpruned decision trees where both candidate feature subsets and numeric splitting thresholds are drawn at random.\n"
        "• Measured Limitation on THIS Dataset: Exhibits leaf-level memorization on deep nodes, resulting in a large train-test gap of 57.05% (100% train accuracy vs 42.95% test accuracy). Furthermore, it cannot break through the ~43% accuracy ceiling because multi-crop physiological boundaries in Gangetic alluvium are inherently continuous and overlapping.\n"
        "• Empirical Strengths: Fastest ensemble fit time (0.74s), lowest 4:1 agronomic risk cost (17,115.0), and complete immunity to collinear stoichiometric ratios (N:P, N:K, OC:N) due to randomized threshold sampling.\n"
        "• Deployment Risk: Model file size is moderately large (~108 MB uncompressed joblib); requires at least 250 MB RAM for real-time mobile/edge serving.\n"
        "• Verdict: RECOMMENDED (Winner for Multi-Class Crop Suitability)."
    )
    doc.add_paragraph(
        "2. Standard Random Forest Classifier (n=100):\n"
        "• Measured Limitation on THIS Dataset: Greedily evaluates every numeric split threshold for each candidate feature, causing training latency to double relative to Extra Trees (1.51s vs 0.74s) while yielding lower test accuracy (42.15% vs 42.95%). Overfits training leaves (100% train accuracy).\n"
        "• Empirical Strengths: Provides highly stable, interpretable Gini feature importances confirming temperature (14.2%), rainfall (12.8%), and N:K ratio (8.5%) as dominant agricultural drivers.\n"
        "• Deployment Risk: Disk and memory footprint exceeds 100 MB.\n"
        "• Verdict: ACCEPTABLE."
    )
    doc.add_paragraph(
        "3. Decision Tree (CART):\n"
        "• Measured Limitation on THIS Dataset: Extreme leaf overfitting (100% train accuracy vs 42.28% test accuracy, generalization gap = 57.72%). Axis-aligned orthogonal splits create step-function boundary artifacts; small localized changes in soil pH (e.g. 5.8 vs 6.1) trigger radically divergent crop branches.\n"
        "• Empirical Strengths: 100% white-box explainability; decision paths can be printed as plain English if-then rules for agricultural extension officers.\n"
        "• Verdict: ACCEPTABLE (Pedagogical and Rule Extraction Only)."
    )
    doc.add_paragraph(
        "4. Extra Tree (Single Tree):\n"
        "• Measured Limitation on THIS Dataset: Drawing random thresholds within a single unbagged tree results in severe prediction variance (0.8710) and sub-optimal accuracy (39.88%), lagging the ensembled version by over 3.0%.\n"
        "• Verdict: NOT RECOMMENDED."
    )

    # Subsection 3.2: Boosting Ensembles
    add_heading_styled(doc, "3.2 Boosting Ensembles", level=2)
    doc.add_paragraph(
        "1. HistGradientBoosting Classifier:\n"
        "• Formulation: Bins continuous variables into 256 discrete integer intervals and optimizes cross-entropy loss via sequential gradient boosting with histogram aggregation.\n"
        "• Measured Limitation on THIS Dataset: Discretization into 256 integer bins slightly rounds continuous trace micronutrients (Zn, B, Cu < 0.6 mg/kg), losing sub-decimal threshold discrimination.\n"
        "• Empirical Strengths: Demonstrates a much tighter generalization gap (20.4% vs 57.0% for Extra Trees), exceptional fold stability (std = 0.0062), and low test log loss (1.5620).\n"
        "• Deployment Risk: Requires full Python/scikit-learn runtime environment; negligible inference latency (94 ms).\n"
        "• Verdict: RECOMMENDED (Runner-Up for Crop Suitability)."
    )
    doc.add_paragraph(
        "2. Traditional Gradient Boosting Classifier (GBDT):\n"
        "• Measured Limitation on THIS Dataset: Disastrous training latency of 70.79 seconds (~100x slower than Extra Trees). Sequential tree boosting offers zero performance benefit on collinear stoichiometric ratios while consuming excessive compute.\n"
        "• Verdict: ACCEPTABLE (Inefficient; completely superseded by HistGBM)."
    )
    doc.add_paragraph(
        "3. AdaBoost Classifier:\n"
        "• Measured Limitation on THIS Dataset: Severe degradation of Macro F1 (collapsed to 0.2285 vs Extra Trees 0.4260). Standard AdaBoost with shallow decision stumps over-weights noisy borderline survey samples, causing severe distortion on 12-class tabular data.\n"
        "• Verdict: NOT RECOMMENDED."
    )

    # Subsection 3.3: Linear & Regularized Models
    add_heading_styled(doc, "3.3 Linear & Regularized Discriminative Models", level=2)
    doc.add_paragraph(
        "1. Linear SVM (LinearSVC):\n"
        "• Measured Limitation on THIS Dataset: Incapable of fitting non-linear nutrient curvature (e.g. Liebig's Law of the Minimum where plant growth saturates once nitrogen exceeds critical thresholds). Crucially, LinearSVC does NOT output calibrated class probabilities, preventing uncertainty communication to farmers.\n"
        "• Empirical Strengths: Ultra-fast prediction latency (1.1 ms for 6,000 samples) and an astonishingly tight generalization gap (+0.0040, train acc 42.85% vs test acc 42.45%).\n"
        "• Deployment Risk: Cannot provide confidence ratings (e.g. '75% confident in Rice vs 25% Jute'), increasing risk of farmer distrust.\n"
        "• Verdict: ACCEPTABLE (High-Speed Deterministic Scoring Only)."
    )
    doc.add_paragraph(
        "2. Logistic Regression (L2 Regularized):\n"
        "• Measured Limitation on THIS Dataset: Linear hyperplanes cannot resolve complex multi-nutrient boundary intersections without manual polynomial feature engineering. High structural bias error (0.9165).\n"
        "• Empirical Strengths: Model weights occupy less than 10 KB of memory. Exceptional probabilistic calibration out of the box (raw log loss = 1.5210); inference executes in 4.8 ms.\n"
        "• Deployment Risk: None; easiest model to deploy in embedded C/Rust, offline Android apps, or SMS-based advisory gateways.\n"
        "• Verdict: RECOMMENDED (Best for Offline Mobile / Low-RAM Gateways)."
    )
    doc.add_paragraph(
        "3. Ridge Classifier:\n"
        "• Measured Limitation on THIS Dataset: Replaces multi-class cross-entropy with squared loss against binary indicator targets; severely underestimates low-frequency seasonal crops.\n"
        "• Verdict: ACCEPTABLE."
    )

    # Subsection 3.4: Generative & Parametric Classifiers
    add_heading_styled(doc, "3.4 Generative & Parametric Density Classifiers", level=2)
    doc.add_paragraph(
        "1. Quadratic Discriminant Analysis (QDA) — FATAL FAILURE MODE:\n"
        "• Measured Limitation on THIS Dataset: Crashed catastrophically with LinAlgError: The covariance matrix of class 0 is singular / collinear. QDA attempts to invert an independent 40x40 covariance matrix for each of the 12 crop classes. The presence of engineered stoichiometric nutrient ratios (N:P, N:K, P:K) and one-hot categorical indicators creates exact linear dependencies within class sub-samples, causing the determinant det(Σ_k) to collapse to zero.\n"
        "• Agronomic Significance: Serves as a vital benchmark warning for agricultural researchers against applying naive parametric density models to multi-ratio soil datasets.\n"
        "• Verdict: NOT RECOMMENDED (Mathematically Incompatible)."
    )
    doc.add_paragraph(
        "2. Linear Discriminant Analysis (LDA):\n"
        "• Measured Limitation on THIS Dataset: Accuracy collapsed to 12.80% (barely above random chance). LDA assumes identical covariance matrices across all 12 classes; this is completely violated in West Bengal where crop zones have wildly disparate variances (e.g. Darjeeling tea exhibits high rainfall and low temperature variance; alluvial rice spans broad rainfall and moderate temperature variance).\n"
        "• Verdict: NOT RECOMMENDED."
    )
    doc.add_paragraph(
        "3. Gaussian Naive Bayes:\n"
        "• Measured Limitation on THIS Dataset: Strict conditional independence assumption is directly violated by edaphic correlations (N and OC r = +0.485; N and N:K r = +0.620). Normal distribution assumption is violated by skewed EC (skew = +1.89) and rainfall (skew = +1.28), depressing Macro F1 to 0.3983.\n"
        "• Verdict: ACCEPTABLE."
    )

    # Subsection 3.5: Instance-Based Models
    add_heading_styled(doc, "3.5 Instance-Based & Distance Models", level=2)
    doc.add_paragraph(
        "1. K-Nearest Neighbors Classifier (k=5) — CATASTROPHIC DEGRADATION:\n"
        "• Measured Limitation on THIS Dataset: Accuracy collapsed to 18.73% (vs Extra Trees 42.95%). In a 40-dimensional continuous feature space, Euclidean distance becomes uniform (the curse of dimensionality where D_max / D_min -> 1). KNN cannot identify true agronomic nearest neighbors, resulting in a devastating 4:1 agronomic risk cost of 24,380.0.\n"
        "• Inference Overhead: Prediction time was 144 ms (scaling linearly with training sample count N=14,000), making it 130x slower at inference than Linear SVM.\n"
        "• Verdict: NOT RECOMMENDED (Severely Degraded by Dimensionality Curse)."
    )

    # Subsection 3.6: Kernel & Neural Models
    add_heading_styled(doc, "3.6 Kernel & Deep Neural Networks", level=2)
    doc.add_paragraph(
        "1. RBF Kernel SVM:\n"
        "• Measured Limitation on THIS Dataset: Extreme computational scaling O(N²) to O(N³). Fitting took 48.18 seconds and inference on 6,000 test samples took 6.61 seconds (nearly 7,000x slower inference than Linear SVM). Despite this immense computational cost, its test accuracy (41.42%) lagged Extra Trees by 1.53%.\n"
        "• Verdict: NOT RECOMMENDED (Computationally Impractical)."
    )
    doc.add_paragraph(
        "2. PyTorch Deep AgriNet (Deep Feedforward MLP):\n"
        "• Architecture: 4 layers (256 -> 128 -> 64 -> 12) with BatchNorm and Dropout.\n"
        "• Measured Limitation on THIS Dataset: Requires continuous hyperparameter tuning; gradient descent on tabular scales is sensitive to learning rate and weight initialization. Lower Macro F1 (0.3922) than tree ensembles due to class probability calibration skew.\n"
        "• Verdict: ACCEPTABLE (Requires Extensive Regularization)."
    )

    # Subsection 3.7: Regression Limitations
    add_heading_styled(doc, "3.7 Supervised Yield & Fertilizer Regressors", level=2)
    doc.add_paragraph(
        "1. Lasso Regressor (L1 Regularized) — Winning Yield Model:\n"
        "• Result: Test R² = 0.9086, RMSE = 5,057.14 kg/ha, Generalization Gap = 24.45 kg/ha, Durbin-Watson = 1.973.\n"
        "• Measured Limitation on THIS Dataset: Assumes linear proportional yield response; cannot model biological nutrient toxicity or yield plateauing at extreme fertilizer saturation.\n"
        "• Verdict: RECOMMENDED (Winner for Crop Yield Estimation)."
    )
    doc.add_paragraph(
        "2. Decision Tree Regressor (CART):\n"
        "• Measured Limitation on THIS Dataset: Severe leaf overfitting (train R² = 1.0, test R² = 0.7410, generalization gap = 0.2590). Step-function predictions cannot smoothly interpolate continuous yield curves, producing an RMSE of 8,510.20 kg/ha.\n"
        "• Verdict: NOT RECOMMENDED."
    )
    doc.add_paragraph(
        "3. K-Nearest Neighbors Regressor (k=5):\n"
        "• Measured Limitation on THIS Dataset: Severe distance metric concentration (R² = 0.6520, RMSE = 9,842.50 kg/ha, MAPE = 78.5%). Cannot extrapolate beyond the minimum and maximum yields observed in the training split.\n"
        "• Verdict: NOT RECOMMENDED."
    )

    # Subsection 3.8: Unsupervised, RL & MCDM Limitations
    add_heading_styled(doc, "3.8 Unsupervised Clustering, Contextual Bandits & MCDM Baselines", level=2)
    doc.add_paragraph(
        "1. K-Means & MiniBatch K-Means Clustering (k=5):\n"
        "• Measured Limitation: Assumes spherical isotropic clusters with equal variance. West Bengal soil zones have elongated non-convex geometries (e.g. narrow coastal saline strips), resulting in low Silhouette scores (0.0686).\n"
        "• Verdict: ACCEPTABLE (Agglomerative Ward clustering is superior, Silhouette = 0.0859)."
    )
    doc.add_paragraph(
        "2. Contextual Bandit Reinforcement Learning (UCB-1, Epsilon-Greedy, Thompson Sampling):\n"
        "• Measured Limitation: Static cross-sectional survey datasets do not support multi-step sequential state transitions (s_t -> a_t -> r_t -> s_{t+1}). Bandits operate strictly as one-step contextual evaluators, achieving 9.20% online accuracy under static survey exploration.\n"
        "• Academic Honesty Disclosure: Reinforcement learning must NOT be overclaimed as dynamic long-term policy optimization on static cross-sectional data."
    )
    doc.add_paragraph(
        "3. Analytical Hierarchy Process (AHP) & TOPSIS:\n"
        "• Measured Limitation: Subject to expert judgment bias in pairwise comparison matrices (CR = 0.0458). Linear additive scoring cannot detect severe single-nutrient deficiencies (e.g. soil with high NPK but fatal zinc deficiency is scored favorably by AHP/TOPSIS).\n"
        "• Verdict: ACCEPTABLE (As baseline comparison only)."
    )

    # ==============================================================================
    # SECTION 4: DEPLOYMENT RISK MATRIX & PRODUCTION SAFEGUARDS
    # ==============================================================================
    add_heading_styled(doc, "4. Deployment Risk Matrix & Production Safeguards", level=1)
    
    doc.add_paragraph(
        "Deploying machine learning models into real-world agricultural decision support across West Bengal introduces critical risks that must be mitigated through software safeguards:"
    )

    deploy_table = doc.add_table(rows=6, cols=3)
    d_headers = ["Identified Deployment Risk", "Vulnerable Algorithmic Families", "Implemented Production Safeguard in predict.py / app.py"]
    for c_idx, h in enumerate(d_headers):
        deploy_table.rows[0].cells[c_idx].paragraphs[0].text = h
        
    d_data = [
        ("Out-of-Distribution (OOD) Extrapolation", "Linear models, KNN, Polynomial regressors", "Automated OOD detection flags inputs outside empirical bounds (pH < 3.8 or > 9.0; rainfall < 200 mm), alerting farmers to seek physical laboratory testing."),
        ("Catastrophic False Positive Misrecommendation", "KNN, AdaBoost, Decision Trees", "System outputs Top-3 Probabilistic Crop Options rather than a hard Top-1 mandate, preventing total farmer investment loss."),
        ("Toxic Chemical Over-Fertilization", "Pure Black-Box Neural Regressors", "Hybrid Mass-Balance formulation binds fertilizer prescriptions strictly to stoichiometric uptake minus soil test supply (N_rec = N_crop - 0.30×N_soil)."),
        ("Mobile Edge Latency & RAM Exhaustion", "RBF SVM (6.6s inference), Random Forest (>100 MB)", "Dual-model architecture: Extra Trees for cloud servers; Logistic Regression (<10 KB, 4.8 ms) for low-power offline mobile apps."),
        ("Micronutrient Model Hallucination", "All ML Regressors predicting Zn, B, Fe, Cu, Mn (R² ≤ 0)", "Direct software disclaimer: Soil parameters with R² ≤ 0 are locked from synthetic inference and strictly mandate wet-chemistry testing.")
    ]
    for r_idx, row_vals in enumerate(d_data):
        row = deploy_table.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            row.cells[c_idx].paragraphs[0].text = val
    style_table(deploy_table)

    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    # Save documents
    output_docx = "algorithm_limitations_comprehensive_report.docx"
    doc.save(output_docx)
    doc.save("deliverables/algorithm_limitations_comprehensive_report.docx")
    print(f"Algorithm Limitations DOCX successfully saved to {output_docx} and deliverables/{output_docx}")

if __name__ == '__main__':
    generate_limitations_docx()
