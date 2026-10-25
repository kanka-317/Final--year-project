import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

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
    # Format Header Row
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
                
    # Format Data Rows
    for i, row in enumerate(table.rows[1:]):
        bg = row_alt_bg if i % 2 == 1 else 'FFFFFF'
        for cell in row.cells:
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=90, bottom=90, left=140, right=140)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
                    run.font.name = 'Calibri'

def create_report():
    doc = Document()
    
    # Page setup - 1 inch margins
    sections = doc.sections
    for section in sections:
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
    title_run = title_p.add_run("Model Training & Empirical Benchmark Report")
    title_run.font.name = 'Calibri'
    title_run.font.size = Pt(24)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(31, 78, 121)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    sub_run = sub_p.add_run("Integrating Soil Fertility, Climate and Geospatial Variables for Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal, India")
    sub_run.font.name = 'Calibri'
    sub_run.font.size = Pt(14)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(89, 89, 89)

    # Metadata callout box
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Academic Affiliation", "Final Year B.Tech Computer Science Project (MAKAUT)"),
        ("Literature Window & Regional Focus", "2022–2026 (West Bengal: Malo & Saha 2025; Singha et al. 2023; Chowdhuri et al. 2026)"),
        ("Dataset & Scale", "20,000 Soil Observations × 23 Districts × 34 Baseline Attributes"),
        ("Algorithms Benchmarked", "50 Algorithms (Supervised, Unsupervised, Contextual Bandits RL, Bio-Inspired GA, MCDM)"),
        ("Validation Protocol", "Leak-Free 70/30 Stratified Holdout, 10-Fold Inner-CV, Global Seed = 42")
    ]
    for row_idx, (k, v) in enumerate(meta_data):
        row = meta_table.rows[row_idx]
        set_cell_background(row.cells[0], 'EBF1F5')
        set_cell_background(row.cells[1], 'F7F9FB')
        p0 = row.cells[0].paragraphs[0]
        r0 = p0.add_run(k)
        r0.font.bold = True
        r0.font.size = Pt(9.5)
        p1 = row.cells[1].paragraphs[0]
        r1 = p1.add_run(v)
        r1.font.size = Pt(9.5)
        set_cell_margins(row.cells[0], 70, 70, 120, 120)
        set_cell_margins(row.cells[1], 70, 70, 120, 120)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ==============================================================================
    # 1. EXECUTIVE SUMMARY
    # ==============================================================================
    h1 = doc.add_heading("1. Executive Summary", level=1)
    h1.style.font.color.rgb = RGBColor(31, 78, 121)
    
    doc.add_paragraph(
        "This research report documents the comprehensive model training, empirical evaluation, and algorithmic diagnostics "
        "conducted for the multi-target agricultural decision support framework across West Bengal, India. By integrating the full suite "
        "of 12 soil physicochemical parameters (pH, EC, OC, N, P, K, S, Fe, Cu, Zn, B, Mn) with meso-climatic variables (temperature, "
        "precipitation, relative humidity) and geospatial coordinates, the project establishes a rigorous empirical benchmark comparing "
        "50 algorithmic implementations across five distinct computational paradigms."
    )

    bullet_p1 = doc.add_paragraph(style='List Bullet')
    r = bullet_p1.add_run("Top-Performing Classifier: ")
    r.bold = True
    bullet_p1.add_run(
        "Extra Trees Classifier (n=100) attained the highest overall generalization performance on the 6,000-sample test partition, "
        "achieving a Macro F1 score of 0.4260, Accuracy of 42.95%, Matthews Correlation Coefficient (MCC) of 0.3777, and Cohen's Kappa of 0.3776 "
        "(compared to the 12-class random chance baseline of 8.33% and Dummy Majority baseline of 1.32% F1)."
    )

    bullet_p2 = doc.add_paragraph(style='List Bullet')
    r = bullet_p2.add_run("Yield Regression Benchmark: ")
    r.bold = True
    bullet_p2.add_run(
        "Lasso Regressor achieved an R² score of 0.9086 with a Root Mean Squared Error (RMSE) of 5,057.14 kg/ha across heterogeneous crops, "
        "confirming that crop yield adheres strongly to linear macro-climatic energy budgets and primary nitrogen-potassium availability."
    )

    bullet_p3 = doc.add_paragraph(style='List Bullet')
    r = bullet_p3.add_run("Nature-Inspired Feature Optimization: ")
    r.bold = True
    bullet_p3.add_run(
        "A binary-encoded Genetic Algorithm (GA) successfully discovered a minimal, non-redundant 24-feature subset from an initial pool of 35 dimensions, "
        "improving Macro F1 (+0.34%) while cutting inference latency by 22.8% and eliminating collinear ratio instabilities."
    )

    bullet_p4 = doc.add_paragraph(style='List Bullet')
    r = bullet_p4.add_run("12-Parameter Error Diagnostics: ")
    r.bold = True
    bullet_p4.add_run(
        "Empirical regressors revealed a profound split in soil property predictability: Potassium (R² = 0.2086, nRMSE = 14.1%), Nitrogen (R² = 0.1383, nRMSE = 14.5%), "
        "Organic Carbon (R² = 0.1009, nRMSE = 16.1%), and pH (R² = 0.0982, nRMSE = 18.0%) exhibit genuine pedo-climatic predictability. In contrast, "
        "micronutrients (Zn, B, Fe, Cu, Mn) and Phosphorus exhibited near-zero or negative R², proving their micro-site, management-driven stochastic nature."
    )

    # ==============================================================================
    # 2. EXPERIMENTAL DESIGN & LEAKAGE PREVENTION
    # ==============================================================================
    h2 = doc.add_heading("2. Experimental Design & Leak-Free Architecture", level=1)
    h2.style.font.color.rgb = RGBColor(31, 78, 121)

    doc.add_paragraph(
        "In strict compliance with non-negotiable scientific protocols, all transformations, scalers, imputers, and feature selectors "
        "were fitted exclusively on the training partition to eliminate data leakage."
    )

    p_leak = doc.add_paragraph()
    r = p_leak.add_run("Data Partitioning Protocol: ")
    r.bold = True
    p_leak.add_run(
        "The master dataset of 20,000 records was partitioned into a 70% Training Split (14,000 samples) and a 30% Testing Split (6,000 samples) "
        "using stratified sampling on the crop target with fixed seed (GLOBAL_SEED = 42). The test partition was locked and evaluated exactly once "
        "after complete model training and hyperparameter selection."
    )

    p_drop = doc.add_paragraph()
    r = p_drop.add_run("Leakage Variable Exclusion: ")
    r.bold = True
    p_drop.add_run(
        "The following attributes were strictly excluded from the predictive feature space:\n"
        "• Administrative keys: record_id, state.\n"
        "• Administrative boundary labels: district (withheld so models generalize on continuous soil/climate features rather than memorizing district names).\n"
        "• Post-harvest production outcomes: yield_kg_ha, production_tonnes, area_ha (using harvest metrics to predict crop represents severe target leakage).\n"
        "• Pre-calculated simulated outcomes: illustrative_SFI_0_100, recommended_N_kg_ha_simulated, recommended_P_kg_ha_simulated, recommended_K_kg_ha_simulated.\n"
        "• Prototyping metadata: data_type, data_source, research_use_warning."
    )

    # ==============================================================================
    # 3. FEATURE ENGINEERING & NATURE-INSPIRED SELECTION
    # ==============================================================================
    h3 = doc.add_heading("3. Domain Feature Engineering & Bio-Inspired Optimization", level=1)
    h3.style.font.color.rgb = RGBColor(31, 78, 121)

    doc.add_paragraph(
        "Informed by regional agronomic literature (Malo & Saha 2025; Singha et al. 2023), 14 domain-engineered variables were derived "
        "to encapsulate stoichiometric nutrient balances, soil reaction deviations, and agro-climatic hydro-thermal indices."
    )

    # Table of Domain Features
    feat_table = doc.add_table(rows=6, cols=3)
    feat_table.rows[0].cells[0].paragraphs[0].text = "Feature Group"
    feat_table.rows[0].cells[1].paragraphs[0].text = "Derived Variables & Equations"
    feat_table.rows[0].cells[2].paragraphs[0].text = "Agronomic Justification"

    feat_data = [
        ("Stoichiometric Ratios", "N:P = N/(P+ε)\nN:K = N/(K+ε)\nP:K = P/(K+ε)\nOC:N = (OC×1000)/(N+ε)", "Captures nutrient balance interactions rather than isolated concentrations; critical for crop nutrient uptake."),
        ("Reaction Deviation", "pH_neutral_deviation = |pH - 7.0|", "Measures soil acidity/alkalinity stress; extreme deviations limit micronutrient bioavailability."),
        ("Micronutrient Flags", "Deficiency flags for Zn (<0.6), B (<0.5), Fe (<4.5), Cu (<0.2), Mn (<2.0 mg/kg)", "Binary flags reflecting ICAR critical deficiency thresholds in West Bengal alluvial/lateritic soils."),
        ("Agro-Climatic Indices", "Hydrothermal = Rain/(Temp+1)\nVapor Pressure Proxy = (Temp×RH)/100", "Quantifies moisture availability vs evaporative thermal demand during crop vegetative phases."),
        ("Calculated SFI Score", "Weighted sum of normalized OC, pH, N, P, K, and micro-sufficiency (0-100)", "Provides continuous composite fertility index following Malo & Saha (2025) multi-attribute weighting.")
    ]
    for row_idx, (g, v, j) in enumerate(feat_data):
        row = feat_table.rows[row_idx + 1]
        row.cells[0].paragraphs[0].text = g
        row.cells[1].paragraphs[0].text = v
        row.cells[2].paragraphs[0].text = j
    style_table(feat_table)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    doc.add_paragraph(
        "Genetic Algorithm (GA) Optimization: Operating with binary chromosomes, tournament selection (k=2), uniform crossover (rate=0.8), "
        "and bit-flip mutation (rate=0.05), the GA evaluated candidate subsets on 3-fold cross-validated Macro F1 with a parsimony bonus. "
        "In 8 generations (25.43s), the GA pruned 11 redundant features, identifying a 24-feature optimal core."
    )

    # Ablation Table
    abl_table = doc.add_table(rows=4, cols=6)
    abl_headers = ["Feature Mode", "Dimensions", "Test Accuracy", "Macro F1", "MCC", "Training Time"]
    for c_idx, h in enumerate(abl_headers):
        abl_table.rows[0].cells[c_idx].paragraphs[0].text = h

    abl_data = [
        ("Raw 12 Soil Parameters", "12", "40.85%", "0.4052", "0.3548", "1.48s"),
        ("All Engineered Features", "35", "42.03%", "0.4174", "0.3676", "2.23s"),
        ("GA-Optimized Subset", "24", "42.21%", "0.4208", "0.3701", "1.72s")
    ]
    for r_idx, row_vals in enumerate(abl_data):
        row = abl_table.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            row.cells[c_idx].paragraphs[0].text = val
    style_table(abl_table)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ==============================================================================
    # 4. SUPERVISED CLASSIFICATION BENCHMARK
    # ==============================================================================
    h4 = doc.add_heading("4. Supervised Classification Benchmark (All Algorithms)", level=1)
    h4.style.font.color.rgb = RGBColor(31, 78, 121)

    doc.add_paragraph(
        "Below is the comprehensive benchmark of all supervised classifiers evaluated on the held-out 30% test partition (6,000 samples). "
        "Models are ranked by Macro F1 score to account for multi-class balance across the 12 agricultural commodities."
    )

    # Classification Master Table
    class_table = doc.add_table(rows=25, cols=7)
    class_headers = ["Rank", "Algorithm", "Category", "Accuracy", "Macro F1", "MCC", "Latency (s)"]
    for c_idx, h in enumerate(class_headers):
        class_table.rows[0].cells[c_idx].paragraphs[0].text = h

    class_data = [
        ("1", "Extra Trees Classifier (n=100)", "Bagging Ensemble", "42.95%", "0.4260", "0.3777", "0.69s"),
        ("2", "Decision Tree (CART)", "Decision Tree", "42.28%", "0.4200", "0.3704", "0.62s"),
        ("3", "HistGradientBoosting", "Boosting Ensemble", "42.03%", "0.4174", "0.3676", "2.23s"),
        ("4", "Linear SVM", "Margin / Kernel", "42.45%", "0.4174", "0.3728", "3.35s"),
        ("5", "Random Forest (n=100)", "Bagging Ensemble", "42.15%", "0.4173", "0.3691", "1.51s"),
        ("6", "Logistic Regression (L2)", "Linear / Probabilistic", "42.32%", "0.4170", "0.3712", "0.48s"),
        ("7", "SGD Classifier", "Linear / Online", "42.15%", "0.4166", "0.3699", "1.57s"),
        ("8", "Voting Ensemble (Soft)", "Ensemble (Voting)", "41.97%", "0.4153", "0.3670", "3.47s"),
        ("9", "MLP Classifier (Scikit-Learn)", "Neural Network", "41.82%", "0.4153", "0.3655", "20.63s"),
        ("10", "Gradient Boosting Classifier", "Boosting Ensemble", "41.83%", "0.4130", "0.3659", "70.79s"),
        ("11", "Ridge Classifier", "Linear Regularized", "42.42%", "0.4122", "0.3735", "0.02s"),
        ("12", "Bagging Classifier", "Bagging Ensemble", "41.63%", "0.4120", "0.3634", "3.64s"),
        ("13", "Bernoulli Naive Bayes", "Probabilistic", "41.68%", "0.4113", "0.3642", "0.02s"),
        ("14", "Perceptron", "Linear Neural", "41.90%", "0.4109", "0.3685", "0.27s"),
        ("15", "RBF SVM", "Kernel Margin", "41.42%", "0.4095", "0.3613", "48.18s"),
        ("16", "Stacking Classifier", "Ensemble (Stacking)", "42.03%", "0.4066", "0.3700", "15.01s"),
        ("17", "Gaussian Naive Bayes", "Probabilistic", "41.43%", "0.3983", "0.3662", "0.01s"),
        ("18", "Extra Tree (Single)", "Decision Tree", "39.88%", "0.3963", "0.3442", "0.03s"),
        ("19", "PyTorch Deep AgriNet", "Deep Learning", "42.20%", "0.3922", "0.3773", "7.33s"),
        ("20", "Nearest Centroid Baseline", "Geometric Baseline", "34.13%", "0.3323", "0.2821", "0.01s"),
        ("21", "AdaBoost Classifier", "Boosting Ensemble", "33.88%", "0.2285", "0.3188", "2.31s"),
        ("22", "KNN Classifier (k=5)", "Instance-Based", "18.73%", "0.1896", "0.1151", "0.15s"),
        ("23", "Linear Discriminant Analysis", "Generative Linear", "12.80%", "0.1244", "0.0485", "0.05s"),
        ("24", "Dummy Majority Baseline", "Trivial Baseline", "8.60%", "0.0132", "0.0000", "0.00s")
    ]
    for r_idx, row_vals in enumerate(class_data):
        row = class_table.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            row.cells[c_idx].paragraphs[0].text = val
    style_table(class_table)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ==============================================================================
    # 5. DETAILED ALGORITHMIC EXPLANATIONS
    # ==============================================================================
    h5 = doc.add_heading("5. In-Depth Algorithmic Explanations & Observed Limitations", level=1)
    h5.style.font.color.rgb = RGBColor(31, 78, 121)

    exp_sections = [
        ("Why Extra Trees Won", 
         "Extremely Randomized Trees (Extra Trees) constructs ensembles by drawing random split thresholds for candidate features rather than "
         "computing exact Gini cut-points. Tabular soil properties exhibit continuous collinearities (e.g. N and OC, or P and pH). The randomized "
         "thresholding mechanism breaks collinear dependencies, resulting in lower variance and superior generalization (0.4260 F1) with rapid training (0.69s)."),
        ("Linear Models Generalization (Logistic Regression & Linear SVM)", 
         "Both models achieved competitive accuracy (~42.4%). This occurs because regional agro-climatic boundaries (e.g. high-rainfall sub-Himalayan zones for Tea "
         "vs arid western plateaus for Millets/Lentils) exhibit strong underlying first-order linear separability across West Bengal districts."),
        ("Deep Learning: PyTorch Deep AgriNet", 
         "The custom 4-layer feedforward network with BatchNorm and Dropout attained strong raw accuracy (42.20%) and MCC (0.3773). However, Macro F1 (0.3922) "
         "lagged tree ensembles. Gradient-based tabular networks are susceptible to feature-scale unbalance and lack the inductive axis-aligned decision split advantages of decision forests."),
        ("Quadratic Discriminant Analysis (QDA) Failure", 
         "QDA failed during training due to a singular covariance matrix. Estimating 12 separate 35×35 covariance matrices caused inversion failure because stoichiometric "
         "ratios and one-hot encoded soil zones introduced exact multicollinearity."),
        ("Degradation of KNN and AdaBoost", 
         "KNN (18.73% accuracy) collapsed due to the curse of dimensionality in 35-dimensional space where Euclidean distances concentrate. "
         "AdaBoost (22.85% F1) suffered because shallow decision stumps over-fit to borderline noisy agricultural data points, distorting multi-class distributions.")
    ]

    for title, desc in exp_sections:
        p = doc.add_paragraph()
        r = p.add_run(f"• {title}: ")
        r.bold = True
        r.font.color.rgb = RGBColor(31, 78, 121)
        p.add_run(desc)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ==============================================================================
    # 6. REGRESSION & 12-PARAMETER ERROR ANALYSIS
    # ==============================================================================
    h6 = doc.add_heading("6. Regression & 12-Parameter Soil Error Analysis", level=1)
    h6.style.font.color.rgb = RGBColor(31, 78, 121)

    doc.add_paragraph(
        "Crop Yield Regression: Lasso Regressor achieved the highest accuracy (R² = 0.9086, RMSE = 5,057.14 kg/ha, MAE = 3,214.20 kg/ha), "
        "closely matched by Ridge (R² = 0.9084) and HistGradientBoosting (R² = 0.8942)."
    )

    doc.add_paragraph(
        "The 12-Parameter Error Benchmark: To determine whether soil parameters can be mapped from remaining inputs, dedicated regressors "
        "were evaluated for each parameter. The results (from HistGradientBoosting) are detailed below:"
    )

    # 12 Soil Parameter Table
    soil_table = doc.add_table(rows=13, cols=6)
    soil_headers = ["Parameter", "Symbol", "R² Score", "RMSE", "Norm. RMSE (nRMSE)", "Predictability Assessment"]
    for c_idx, h in enumerate(soil_headers):
        soil_table.rows[0].cells[c_idx].paragraphs[0].text = h

    soil_data = [
        ("Available Potassium", "K", "0.2086", "44.86 kg/ha", "14.13%", "Moderate / Highest in suite"),
        ("Available Nitrogen", "N", "0.1383", "56.12 kg/ha", "14.52%", "Moderate"),
        ("Organic Carbon", "OC", "0.1009", "0.221 %", "16.09%", "Moderate"),
        ("Soil pH", "pH", "0.0982", "0.811", "18.01%", "Moderate"),
        ("Zinc", "Zn", "-0.0017", "0.342 mg/kg", "15.22%", "Extremely Low / Stochastic"),
        ("Available Phosphorus", "P", "-0.0028", "11.83 kg/ha", "15.35%", "Extremely Low / Chemically Fixed"),
        ("Iron", "Fe", "-0.0028", "6.86 mg/kg", "15.74%", "Extremely Low / Stochastic"),
        ("Available Sulphur", "S", "-0.0033", "7.88 mg/kg", "15.76%", "Extremely Low / Stochastic"),
        ("Copper", "Cu", "-0.0038", "0.394 mg/kg", "15.88%", "Extremely Low / Stochastic"),
        ("Manganese", "Mn", "-0.0052", "2.96 mg/kg", "15.51%", "Extremely Low / Stochastic"),
        ("Boron", "B", "-0.0025", "0.248 mg/kg", "16.99%", "Extremely Low / Stochastic"),
        ("Electrical Conductivity", "EC", "-0.0040", "0.387 dS/m", "13.19%", "Extremely Low / Highly Localized")
    ]
    for r_idx, row_vals in enumerate(soil_data):
        row = soil_table.rows[r_idx + 1]
        for c_idx, val in enumerate(row_vals):
            row.cells[c_idx].paragraphs[0].text = val
    style_table(soil_table)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    doc.add_paragraph(
        "Agronomic Insight: Nitrogen, Potassium, and Organic Carbon are tied to broad biological and climatic processes (temperature-driven "
        "decomposition and rainfall weathering). Conversely, Phosphorus (subject to rapid mineral adsorption) and micronutrients (governed by "
        "localized organic amendments and micro-redox conditions) cannot be predicted from macro-geospatial factors. Point-of-care laboratory testing remains mandatory."
    )

    # ==============================================================================
    # 7. UNSUPERVISED, SEMI-SUPERVISED & REINFORCEMENT LEARNING
    # ==============================================================================
    h7 = doc.add_heading("7. Unsupervised, Semi-Supervised & Contextual Bandits", level=1)
    h7.style.font.color.rgb = RGBColor(31, 78, 121)

    doc.add_paragraph(
        "• Unsupervised Clustering: Agglomerative Hierarchical Clustering (Ward linkage) achieved the best natural grouping (Silhouette = 0.0859, "
        "Calinski-Harabasz = 148.80, Davies-Bouldin = 2.4676), demarcating five distinct pedo-climatic agro-zones (Sub-Himalayan Terai, Coastal Saline, "
        "Western Lateritic, Old Alluvial, and New Riverine Alluvium)."
    )

    doc.add_paragraph(
        "• Semi-Supervised Learning (50% Masked Labels): Self-Training Classifier (HistGBM base) recovered 96.1% of fully supervised accuracy (40.82% Accuracy, "
        "0.4045 Macro F1), proving viable for semi-automated extension when soil tests lack crop suitability labels."
    )

    doc.add_paragraph(
        "• Reinforcement Learning (Contextual Bandits): Evaluated across 3,000 online decision steps, Upper Confidence Bound (UCB-1) achieved an average "
        "reward (accuracy) of 9.20% (276 rewards), outperforming Epsilon-Greedy (8.83%) and Thompson Sampling (8.07%). Static cross-sectional survey data "
        "justifies a single-step contextual bandit formulation rather than full sequential MDPs."
    )

    doc.add_paragraph(
        "• Multi-Criteria Decision Making (AHP & TOPSIS): AHP expert weights for the 6 primary criteria were verified with Consistency Ratio CR = 0.0458 < 0.10 "
        "(OC: 30.12%, N: 30.12%, P: 11.85%, K: 11.85%, pH: 11.85%, EC: 4.21%). TOPSIS proximity scoring correlated strongly with high-demand commercial crops."
    )

    # ==============================================================================
    # 8. PRODUCTION INFERENCE & SUMMARY
    # ==============================================================================
    h8 = doc.add_heading("8. Production Prediction System & Research Deliverables", level=1)
    h8.style.font.color.rgb = RGBColor(31, 78, 121)

    doc.add_paragraph(
        "The complete inference pipeline has been encapsulated into an exportable, leak-free deployment function `predict_crop_and_fertilizer()`. "
        "Given 12 soil parameters and climatic conditions, it returns:"
    )

    doc.add_paragraph(
        "1. Top-3 Suitable Crop Ranking with calibrated predictive confidence probabilities.\n"
        "2. Soil Fertility Index (SFI) score and class (High, Moderate, Low).\n"
        "3. Limiting Nutrient Identification (Liebig's Law of the Minimum).\n"
        "4. Site-Specific Fertilizer Doses (N, P₂O₅, K₂O in kg/ha) computed as Crop Removal Requirement minus Soil Test Contribution.\n"
        "5. Out-of-Distribution (OOD) Uncertainty Warning when input variables exceed training domain boundaries."
    )

    # Save documents
    out_dir_file = os.path.join('deliverables', 'model_training_report.docx')
    root_file = 'model_training_report.docx'
    doc.save(out_dir_file)
    doc.save(root_file)
    print(f"Report successfully saved to {out_dir_file} and {root_file}")

if __name__ == '__main__':
    create_report()
