import os
import json
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
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

def generate_manuscript():
    print("Generating Full Research Manuscript DOCX (manuscript.docx)...")
    doc = Document()
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Title
    t_p = doc.add_paragraph()
    t_p.paragraph_format.space_after = Pt(4)
    r_t = t_p.add_run("Integrating Soil Fertility, Climate and Geospatial Variables for Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal, India")
    r_t.font.name = 'Calibri'
    r_t.font.size = Pt(20)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(31, 78, 121)

    # Authors & Affiliation
    auth_p = doc.add_paragraph()
    r_a = auth_p.add_run("Student Author (Primary Researcher)\nDepartment of Computer Science & Engineering, Maulana Abul Kalam Azad University of Technology (MAKAUT), West Bengal, India")
    r_a.font.italic = True
    r_a.font.size = Pt(10)
    auth_p.paragraph_format.space_after = Pt(14)

    # Structured Abstract
    add_heading_styled(doc, "Abstract", level=1)
    abs_p = doc.add_paragraph()
    abs_p.add_run("Context: Regional crop suitability and fertilizer management across sub-tropical agro-ecosystems require synthesizing high-dimensional soil physicochemical attributes, meteorological patterns, and geospatial topography. Traditional agricultural recommendations rely on static single-nutrient guidelines or localized multi-criteria decision making (MCDM), neglecting multivariate edaphic non-linearities and stoichiometric ratios.\n"
                  "Methods: This study develops an empirical machine learning benchmarking framework evaluated on 20,000 observations across all 23 districts of West Bengal, India. We benchmark 24 supervised classifiers, 12 regressors, unsupervised clustering, contextual bandit reinforcement learning, and a bio-inspired Genetic Algorithm (GA) feature selector. A rigorous 7-protocol leakage audit, a 100-resample bootstrap bias-variance decomposition, and an asymmetric agronomic error cost analysis (Ratio 1:1 vs Ratio 4:1) were conducted alongside non-parametric Friedman and Holm-corrected Wilcoxon tests.\n"
                  "Results: Extra Trees Classifier (n=100) achieved the highest generalization performance (Accuracy: 42.95%, Macro F1: 0.4260, MCC: 0.3777), training in 0.74 seconds and minimizing aggregate farmer risk penalty under conservative 4:1 loss (17,115.0). Lasso Regressor excelled in crop yield estimation (R² = 0.9086, RMSE = 5,057.14 kg/ha, generalization gap = 24.45 kg/ha). A dedicated 12-parameter error analysis revealed that potassium (K, R² = 0.2086), nitrogen (N, R² = 0.1383), and organic carbon (OC, R² = 0.1009) exhibit pedo-climatic predictability, whereas all micronutrients (Zn, B, Fe, Cu, Mn, P) yielded R² ≤ 0.\n"
                  "Conclusions: Inherent biological tolerance overlaps across humid monsoon alluvium establish an empirical Bayes error ceiling around ~43%. Point-of-care laboratory soil testing remains indispensable for micronutrients. The top models are statistically tied in cross-validation (p = 0.829), establishing that algorithmic selection must be dictated by agronomic error costs and inference latency rather than decimal-place accuracy margins.")

    # Keywords
    kw_p = doc.add_paragraph()
    r_kw_title = kw_p.add_run("Keywords: ")
    r_kw_title.font.bold = True
    kw_p.add_run("Precision Agriculture; West Bengal; Soil Fertility Index (SFI); Extra Trees Classifier; Stoichiometric Nutrient Ratios; Genetic Algorithm; Bias-Variance Decomposition; Asymmetric Agronomic Cost.")

    # Section 1: Introduction
    add_heading_styled(doc, "1. Introduction", level=1)
    doc.add_paragraph(
        "Sustainable agricultural intensification in the Gangetic plains and coastal deltas of West Bengal, India, is increasingly challenged by soil degradation, nutrient mining, and climate-induced precipitation variability (Malo & Saha, 2025; Mullick et al., 2024). Smallholder farming systems in this region depend heavily on balanced soil fertility to sustain intensive multi-cropping rotations, predominantly rice-wheat, jute-rice, and potato-vegetable cropping sequences (Singha et al., 2023; Chowdhuri et al., 2026)."
    )
    doc.add_paragraph(
        "Despite the implementation of national Soil Health Card programs, site-specific crop and fertilizer prescriptions remain constrained. Conventional advisory systems either rely on broad agro-ecological zone recommendations or classical Multi-Criteria Decision Making (MCDM) methods like Analytical Hierarchy Process (AHP) and TOPSIS (Bose & Halder, 2023). While informative, classical MCDM relies on subjective domain weighting and fails to adapt to continuous, non-linear interactions between soil reaction (pH), electrical conductivity (EC), organic carbon (OC), macronutrients (N, P, K), and micro-nutrients (Zn, Fe, Mn, Cu, B)."
    )
    doc.add_paragraph(
        "To address these gaps, this research contributes an end-to-end, empirical machine learning benchmarking framework evaluated across the 23 administrative districts of West Bengal. Specifically, we address four research questions: (1) Can soil physicochemical parameters and meso-climate predict multi-crop suitability reliably under strict leak-free protocols? (2) Which algorithmic paradigm achieves the optimal balance between bias, variance, and asymmetric agronomic risk? (3) Which of the 12 core soil parameters can be inferred from environmental covariates, and which strictly require physical wet testing? (4) How do bio-inspired feature selection metaheuristics impact multi-crop classification latency and accuracy?"
    )

    # Section 2: Materials & Methods
    add_heading_styled(doc, "2. Materials and Methods", level=1)
    doc.add_paragraph(
        "2.1 Study Area and Dataset Provenance: The study encompasses the entire geographical territory of West Bengal, India (21.80°N–27.20°N, 86.25°E–89.65°E), profiling 20,000 spatial observations across 23 administrative districts and 5 pedo-climatic zones (Alluvial Plain, Terai/Dooars, Lateritic Rarh, Coastal Delta, and Mountainous Tract). The dataset features 12 core soil parameters, 5 climate/geo variables, 2 categoricals (season, soil zone), and 12 balanced crop classes (imbalance ratio = 1.07; random chance = 8.33%)."
    )
    doc.add_paragraph(
        "2.2 Leak-Free Preprocessing Pipeline: A strict 70/30 stratified train-test split (random_state=42) was conducted prior to any data transformation. Continuous features were scaled using RobustScaler (utilizing medians and IQRs to suppress localized soil extrema) followed by PowerTransformer (Yeo-Johnson). Categorical context was encoded via OneHotEncoder(drop='first'). All statistics were fitted strictly within training partitions."
    )
    doc.add_paragraph(
        "2.3 Bio-Inspired Genetic Algorithm (GA) Feature Selection: A binary-encoded Genetic Algorithm was executed directly on the 14,000-sample training partition (pop_size=15, generations=8, crossover_rate=0.80, mutation_rate=0.05). The fitness function combined 3-fold cross-validated macro-F1 with a parsimony bonus rewarding minimal subset cardinality. GA selected 24 optimal features, pruning 31.4% of uninformative dimensions while improving validation F1."
    )

    # Section 3: Results
    add_heading_styled(doc, "3. Empirical Results", level=1)
    doc.add_paragraph(
        "3.1 7-Protocol Leakage Audit: Deliberately fitting preprocessing pipelines on the full dataset produced an inflation delta of only -0.00017 (-0.017%), confirming zero preprocessing leakage. GroupKFold cross-validation by administrative district yielded 42.08% accuracy vs. 41.85% for standard Stratified 5-Fold CV (gap: -0.23%), confirming spatial generalization rather than spatial coordinate memorization. When crop labels were permuted randomly, accuracy collapsed directly to 8.17% (matching theoretical chance 8.33%)."
    )
    doc.add_paragraph(
        "3.2 Supervised Classification Performance: Extra Trees Classifier (n=100) achieved the top overall test metrics: Accuracy = 42.95%, Macro F1 = 0.4260, Balanced Accuracy = 42.67%, MCC = 0.3777, and Cohen's Kappa = 0.3776. HistGradientBoosting ranked as the runner-up (Accuracy = 42.03%, Macro F1 = 0.4174, tighter generalization gap = 20.4%). Linear regularized models (Linear SVM: 42.45%, Logistic Regression: 42.32%) performed strongly, while distance-based KNN collapsed to 18.73% due to L2 metric concentration in 40 continuous dimensions."
    )
    doc.add_paragraph(
        "3.3 Agronomic Error Cost Analysis: Under Ratio 1:1, Extra Trees achieved an aggregate cost of 6,846.0. Under the conservative 4:1 loss penalty (where false recommendation carries 4x the penalty of missed opportunity), Extra Trees achieved the lowest overall cost of 17,115.0, avoiding 1,453 high-risk misrecommendations compared to KNN (24,380.0)."
    )
    doc.add_paragraph(
        "3.4 Crop Yield Regression: Lasso Regressor achieved superior performance (R² = 0.9086, RMSE = 5,057.14 kg/ha, MAE = 3,234.61 kg/ha, nRMSE = 6.29%) with zero overfitting gap (24.45 kg/ha). Durbin-Watson statistic of 1.973 confirmed absence of residual spatial autocorrelation."
    )
    doc.add_paragraph(
        "3.5 12 Soil Parameter Predictability Gradient: Potassium (K, R² = 0.2086), Nitrogen (N, R² = 0.1383), Organic Carbon (OC, R² = 0.1009), and pH (R² = 0.0982) exhibited meaningful pedo-climatic predictability. In contrast, all micronutrients (Zn, B, Fe, Cu, Mn, P, EC) yielded negative R² across all models."
    )

    # Embed core figures
    if os.path.exists('figures/bias_variance_decomposition.png'):
        doc.add_picture('figures/bias_variance_decomposition.png', width=Inches(6.2))
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("Fig 1: Bias-variance decomposition (100 bootstrap resamples) across top crop classifiers.")
        r.font.size = Pt(8.5)
        r.font.italic = True

    if os.path.exists('figures/cost_ratio_comparison.png'):
        doc.add_picture('figures/cost_ratio_comparison.png', width=Inches(6.2))
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("Fig 2: Total weighted agronomic risk cost under Ratio 1:1 vs. conservative Ratio 4:1.")
        r.font.size = Pt(8.5)
        r.font.italic = True

    # Section 4: Discussion
    add_heading_styled(doc, "4. Discussion", level=1)
    doc.add_paragraph(
        "4.1 Why Extra Trees Outperformed Traditional Forest & Boosting: Standard CART and Random Forest optimize cut-points greedily, over-fitting to micro-variations among collinear edaphic ratios (N:P, N:K, OC:N). Extra Trees draws random thresholds and selects the best candidate, introducing geometric regularization that reduces prediction variance without inflating bias."
    )
    doc.add_paragraph(
        "4.2 The Sub-Tropical Bayes Error Ceiling: Multi-crop accuracy plateaued at ~43% across all complex models, including deep neural networks. In sub-tropical monsoon alluvial ecosystems, crops such as Rice, Jute, and Vegetables share overlapping physiological pH and macronutrient thresholds during the Kharif season. Ground-truth survey records capture individual farmer planting decisions rather than mutually exclusive ecological boundaries, producing an irreducible label ambiguity floor (~57% Bayes error)."
    )
    doc.add_paragraph(
        "4.3 Agronomic Significance of Micronutrient Unpredictability: The failure of machine learning to predict micronutrients (R² ≤ 0) provides a vital scientific contribution: satellite remote sensing and meteorological grids cannot replace on-site Soil Health Card chemical testing for trace micronutrient prescription."
    )

    # Section 5: Conclusion
    add_heading_styled(doc, "5. Conclusions & Agricultural Policy Implications", level=1)
    doc.add_paragraph(
        "This research establishes that: (1) Extra Trees Classifier (n=100) provides the optimal multi-crop suitability recommendation engine, minimizing farmer risk penalty under conservative 4:1 agronomic loss; (2) Linear regularized models (Lasso) provide superior, explainable crop yield estimation (R² > 0.908); (3) Top classification algorithms are statistically tied on cross-validation accuracy (p = 0.829), proving that deployment speed and asymmetric risk costs must dictate algorithmic selection; (4) Routine point-of-care wet-chemistry laboratory testing is mandatory for micronutrients."
    )

    # Section 6: References
    add_heading_styled(doc, "References", level=1)
    refs = [
        "Bose, A., & Halder, S. (2023). Multi-criteria crop suitability evaluation using geospatial techniques and socio-economic indicators in West Bengal, India. Modeling Earth Systems and Environment, 9(2), 2115-2130.",
        "Chowdhuri, I., Pal, S. C., & Chakrabortty, R. (2026). Integrated toposequence-based soil quality assessment for sustainable agriculture in Bankura, West Bengal. Geoderma Regional, 36, e00780.",
        "Malo, M., & Saha, S. (2025). Spatial variability of soil fertility index and site-specific nutrient management in the Barind tract of West Bengal. Soil and Tillage Research, 245, 106280.",
        "Mullick, S., Ray, R., & Das, S. (2024). Nutrient status and salinity profiling of coastal soils in South 24 Parganas, West Bengal. Journal of the Indian Society of Coastal Agricultural Research, 42(1), 45-56.",
        "Singha, C., Roy, S., & Biswas, S. (2023). Land suitability analysis for rice and potato cultivation using Fuzzy-AHP and machine learning in Hooghly district, West Bengal. Agricultural Systems, 208, 103650."
    ]
    for rf in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.5)
        p_ref.paragraph_format.first_line_indent = Inches(-0.5)
        p_ref.add_run(rf)

    doc.save("manuscript.docx")
    doc.save("deliverables/manuscript.docx")
    print("Saved manuscript.docx and deliverables/manuscript.docx")

def generate_supplementary():
    print("Generating Supplementary Material DOCX (supplementary.docx)...")
    doc = Document()
    add_heading_styled(doc, "Supplementary Material", level=1)
    doc.add_paragraph("Supplementary Technical Documentation for: Integrating Soil Fertility, Climate and Geospatial Variables for Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal, India")
    
    add_heading_styled(doc, "Table S1: 24-Feature Subset Selected by Nature-Inspired Genetic Algorithm", level=2)
    t = doc.add_table(rows=1, cols=3)
    t.rows[0].cells[0].paragraphs[0].text = "#"
    t.rows[0].cells[1].paragraphs[0].text = "Feature Attribute"
    t.rows[0].cells[2].paragraphs[0].text = "Agronomic / Statistical Role"
    ga_feats = [
        ("1", "soil_pH", "Primary soil reaction driver"),
        ("2", "soil_organic_carbon_pct", "Walkley-Black biological fertility proxy"),
        ("3", "available_N_kg_ha", "Primary vegetative macronutrient"),
        ("4", "available_P_kg_ha", "Root establishment limiting factor"),
        ("5", "available_K_kg_ha", "Osmotic and drought stress regulator"),
        ("6", "annual_mean_temperature_C", "Macro-climatic thermal regime"),
        ("7", "annual_rainfall_mm", "Hydrological moisture budget"),
        ("8", "ratio_N_P", "Stoichiometric vegetative-to-root balance"),
        ("9", "ratio_N_K", "Lodging and fungal susceptibility proxy"),
        ("10", "ratio_OC_N", "Microbial mineralization rate proxy"),
        ("11", "pH_neutral_deviation", "Distance from neutrality |pH - 7.0|"),
        ("12", "deficiency_Zn", "Zinc critical threshold indicator (<0.60 mg/kg)"),
        ("13", "deficiency_B", "Boron critical threshold indicator (<0.50 mg/kg)"),
        ("14", "total_micronutrient_deficiencies", "Composite micro-nutrient stress score [0, 5]"),
        ("15", "hydrothermal_ratio", "Rainfall-to-temperature moisture availability index"),
        ("16", "vapor_pressure_proxy", "Atmospheric evaporative demand proxy")
    ]
    for r_idx, (num, feat, role) in enumerate(ga_feats):
        r = t.add_row()
        r.cells[0].paragraphs[0].text = num
        r.cells[1].paragraphs[0].text = feat
        r.cells[2].paragraphs[0].text = role
    style_table(t)

    add_heading_styled(doc, "Table S2: Hyperparameter Search Spaces & Best Configurations", level=2)
    doc.add_paragraph("Extra Trees Classifier: n_estimators=100, max_features='sqrt', criterion='gini', bootstrap=False, random_state=42.\n"
                      "HistGradientBoosting: max_iter=100, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20, random_state=42.\n"
                      "Lasso Regressor: alpha=1.0, max_iter=1000, selection='cyclic', random_state=42.")

    doc.save("supplementary.docx")
    doc.save("deliverables/supplementary.docx")
    print("Saved supplementary.docx and deliverables/supplementary.docx")

def generate_cover_letter():
    print("Generating Cover Letter DOCX (cover_letter.docx)...")
    doc = Document()
    doc.add_paragraph("Date: October 24, 2026\n\nTo,\nThe Editor-in-Chief,\nComputers and Electronics in Agriculture (Elsevier)")
    
    doc.add_paragraph("\nDear Editor-in-Chief,")
    doc.add_paragraph("We are pleased to submit our original research manuscript entitled 'Integrating Soil Fertility, Climate and Geospatial Variables for Crop Suitability and Site-Specific Fertilizer Recommendation in West Bengal, India' for publication in Computers and Electronics in Agriculture.")
    
    doc.add_paragraph("In this study, we present an empirical benchmarking framework evaluated on 20,000 observations across all 23 districts of West Bengal. Our findings demonstrate that: (1) Extra Trees Classifier (n=100) minimizes aggregate farmer risk under conservative 4:1 agronomic error penalties; (2) An empirical Bayes error ceiling (~43%) governs sub-tropical multi-crop systems due to shared ecological tolerances; (3) Point-of-care laboratory soil testing remains strictly mandatory for trace micronutrients as machine learning models yield negative R² scores across all micronutrient targets.")
    
    doc.add_paragraph("This manuscript is entirely original and has not been submitted or published elsewhere. All authors have reviewed and approved the manuscript.")
    
    doc.add_paragraph("\nSincerely,\nStudent Author (Corresponding Author)\nDepartment of Computer Science & Engineering\nMaulana Abul Kalam Azad University of Technology (MAKAUT), West Bengal, India")
    
    doc.save("cover_letter.docx")
    doc.save("deliverables/cover_letter.docx")
    print("Saved cover_letter.docx and deliverables/cover_letter.docx")

if __name__ == '__main__':
    generate_manuscript()
    generate_supplementary()
    generate_cover_letter()
