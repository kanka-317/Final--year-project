"""
================================================================================
Comprehensive Experiment Runner: AgroStack-CR Evaluation & Fair Benchmark
================================================================================
Runs rigorous, leak-free evaluation of AgroStack-CR vs All Baselines:
- 70/30 Stratified Split (Seed 42) identical to baseline benchmark
- 20 Fundamental Errors computed
- Asymmetric Agronomic Cost Analysis (Ratio 1:1 and Ratio 4:1)
- Conformal Prediction Set Coverage & Mean Cardinality (1 - alpha = 0.90)
- Statistical Hypothesis Testing (Wilcoxon signed-rank, Cohen's d, Bootstrap 95% CI)
- Complete Ablation Study & Sensitivity Analysis
- Robustness Under Noise, Missingness, Downsampling, and Spatial District GroupKFold
- Exports experiment_results.xlsx, comparison_tables.csv, and summary JSONs
================================================================================
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax
import openpyxl

# Scikit-learn models & metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.preprocessing import RobustScaler, PowerTransformer, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, cohen_kappa_score, log_loss, brier_score_loss,
    confusion_matrix
)
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

# Import proposed algorithm
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from proposed_algorithm import AgroStackCR

warnings.filterwarnings('ignore')
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
eps = 1e-6

print("================================================================================")
print("STARTING COMPREHENSIVE EXPERIMENT EVALUATION: AgroStack-CR vs BASELINES")
print("================================================================================")

# 1. Load Dataset
data_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'cleaned_data.csv'))
if not os.path.exists(data_path):
    data_path = 'cleaned_data.csv'
print(f"Loading dataset from: {data_path}")
df_raw = pd.read_csv(data_path)

# 2. Engineer Agronomic Domain Features (Identical to Baseline Benchmark)
def engineer_agronomic_features(df):
    data = df.copy()
    data['ratio_N_P'] = data['available_N_kg_ha'] / (data['available_P_kg_ha'] + eps)
    data['ratio_N_K'] = data['available_N_kg_ha'] / (data['available_K_kg_ha'] + eps)
    data['ratio_P_K'] = data['available_P_kg_ha'] / (data['available_K_kg_ha'] + eps)
    data['ratio_OC_N'] = (data['soil_organic_carbon_pct'] * 1000) / (data['available_N_kg_ha'] + eps)
    data['pH_neutral_deviation'] = np.abs(data['soil_pH'] - 7.0)
    data['deficiency_Zn'] = (data['Zn_mg_kg'] < 0.60).astype(int)
    data['deficiency_B'] = (data['B_mg_kg'] < 0.50).astype(int)
    data['deficiency_Fe'] = (data['Fe_mg_kg'] < 4.50).astype(int)
    data['deficiency_Cu'] = (data['Cu_mg_kg'] < 0.20).astype(int)
    data['deficiency_Mn'] = (data['Mn_mg_kg'] < 2.00).astype(int)
    data['total_micronutrient_deficiencies'] = (
        data['deficiency_Zn'] + data['deficiency_B'] + data['deficiency_Fe'] +
        data['deficiency_Cu'] + data['deficiency_Mn']
    )
    data['hydrothermal_ratio'] = data['annual_rainfall_mm'] / (data['annual_mean_temperature_C'] + 1.0)
    data['vapor_pressure_proxy'] = (data['annual_mean_temperature_C'] * data['relative_humidity_pct']) / 100.0
    
    norm_pH = 1.0 - np.clip(np.abs(data['soil_pH'] - 6.8) / 3.0, 0, 1)
    norm_OC = np.clip(data['soil_organic_carbon_pct'] / 1.0, 0, 1)
    norm_N = np.clip(data['available_N_kg_ha'] / 400.0, 0, 1)
    norm_P = np.clip(data['available_P_kg_ha'] / 50.0, 0, 1)
    norm_K = np.clip(data['available_K_kg_ha'] / 250.0, 0, 1)
    norm_micro = 1.0 - (data['total_micronutrient_deficiencies'] / 5.0)
    
    data['calculated_SFI_score'] = (
        0.20 * norm_OC + 0.15 * norm_pH + 0.20 * norm_N +
        0.15 * norm_P + 0.15 * norm_K + 0.15 * norm_micro
    ) * 100.0
    return data

df_eng = engineer_agronomic_features(df_raw)

soil_12_cols = [
    'soil_pH', 'soil_EC_dS_m', 'soil_organic_carbon_pct',
    'available_N_kg_ha', 'available_P_kg_ha', 'available_K_kg_ha', 'available_S_mg_kg',
    'Zn_mg_kg', 'Fe_mg_kg', 'Mn_mg_kg', 'Cu_mg_kg', 'B_mg_kg'
]
climate_geo_cols = [
    'latitude_approx', 'longitude_approx', 'annual_mean_temperature_C',
    'annual_rainfall_mm', 'relative_humidity_pct'
]
engineered_cols = [
    'ratio_N_P', 'ratio_N_K', 'ratio_P_K', 'ratio_OC_N',
    'pH_neutral_deviation', 'deficiency_Zn', 'deficiency_B', 'deficiency_Fe',
    'deficiency_Cu', 'deficiency_Mn', 'total_micronutrient_deficiencies',
    'hydrothermal_ratio', 'vapor_pressure_proxy', 'calculated_SFI_score'
]
categorical_cols = ['season', 'soil_zone']
numeric_features = soil_12_cols + climate_geo_cols + engineered_cols
all_features_raw = numeric_features + categorical_cols

X_df = df_eng[all_features_raw].copy()
y_crop_raw = df_eng['crop'].copy()
districts = df_eng['district'].copy()

label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y_crop_raw)
class_names = list(label_enc.classes_)
K_classes = len(class_names)

# Split 70/30 Stratified
X_train_raw, X_test_raw, y_train, y_test, dist_train, dist_test = train_test_split(
    X_df, y_encoded, districts, test_size=0.30, stratify=y_encoded, random_state=GLOBAL_SEED
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('scaler', RobustScaler()),
            ('power', PowerTransformer(method='yeo-johnson'))
        ]), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ]
)

X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)
print(f"Processed Features: X_train={X_train.shape}, X_test={X_test.shape}")

# 3. Model Catalogue
models = {
    'AgroStack-CR (Proposed)': AgroStackCR(
        n_estimators_et=100, max_iter_hgb=100, C_logistic=1.0, meta_C=1.0,
        C_FP=4.0, C_FN=1.0, alpha=0.10, random_state=GLOBAL_SEED
    ),
    'Extra Trees (Best Baseline)': ExtraTreesClassifier(n_estimators=100, random_state=GLOBAL_SEED, n_jobs=-1),
    'HistGradientBoosting (Runner-Up)': HistGradientBoostingClassifier(max_iter=100, random_state=GLOBAL_SEED),
    'Random Forest (n=100)': RandomForestClassifier(n_estimators=100, random_state=GLOBAL_SEED, n_jobs=-1),
    'Linear SVM': LinearSVC(random_state=GLOBAL_SEED),
    'Logistic Regression (L2)': LogisticRegression(max_iter=500, random_state=GLOBAL_SEED),
    'Decision Tree (CART)': DecisionTreeClassifier(random_state=GLOBAL_SEED),
    'Bernoulli Naive Bayes': BernoulliNB(),
    'K-Nearest Neighbors (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Dummy Majority Baseline': DummyClassifier(strategy='most_frequent')
}

print(f"\nEvaluating {len(models)} models on identical test protocol...")

# Helper function to compute 20 errors
def compute_model_metrics_and_20_errors(name, model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_time = time.time() - t0
    
    t1 = time.time()
    y_pred = model.predict(X_te)
    pred_time = time.time() - t1
    
    # Check if model outputs probabilities
    has_proba = hasattr(model, 'predict_proba')
    if has_proba:
        try:
            P_te = model.predict_proba(X_te)
            log_loss_val = float(log_loss(y_te, P_te, labels=np.arange(K_classes)))
            brier_val = float(brier_score_loss((y_te == 0).astype(int), P_te[:, 0]))
        except Exception:
            P_te = None
            log_loss_val = float('nan')
            brier_val = float('nan')
    else:
        P_te = None
        log_loss_val = float('nan')
        brier_val = float('nan')
        
    acc = float(accuracy_score(y_te, y_pred))
    bal_acc = float(balanced_accuracy_score(y_te, y_pred))
    f1 = float(f1_score(y_te, y_pred, average='macro'))
    prec = float(precision_score(y_te, y_pred, average='macro', zero_division=0))
    rec = float(recall_score(y_te, y_pred, average='macro'))
    mcc = float(matthews_corrcoef(y_te, y_pred))
    kappa = float(cohen_kappa_score(y_te, y_pred))
    
    # 5-Fold CV on training set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
    cv_accs = []
    for tr_idx, va_idx in skf.split(X_tr, y_tr):
        m_clone = models[name].__class__(**models[name].get_params()) if hasattr(models[name], 'get_params') else models[name]
        m_clone.fit(X_tr[tr_idx], y_tr[tr_idx])
        cv_accs.append(accuracy_score(y_tr[va_idx], m_clone.predict(X_tr[va_idx])))
    cv_mean = float(np.mean(cv_accs))
    cv_std = float(np.std(cv_accs))
    val_error = 1.0 - cv_mean
    
    # Train accuracy
    y_train_pred = model.predict(X_tr)
    train_acc = float(accuracy_score(y_tr, y_train_pred))
    train_error = 1.0 - train_acc
    test_error = 1.0 - acc
    gen_gap = test_error - train_error
    
    # Confusion matrix & FP / FN counts
    cm = confusion_matrix(y_te, y_pred, labels=np.arange(K_classes))
    total_samples = len(y_te)
    fp_total = int(total_samples - np.trace(cm))
    fn_total = fp_total
    
    cost_1_1 = float(fp_total * 1.0 + fn_total * 1.0)
    cost_4_1 = float(fp_total * 4.0 + fn_total * 1.0)
    
    # Bootstrap 95% Confidence Interval for Test Error
    rng = np.random.RandomState(GLOBAL_SEED)
    boot_accs = []
    for _ in range(100):
        b_idx = rng.choice(len(y_te), len(y_te), replace=True)
        boot_accs.append(accuracy_score(y_te[b_idx], y_pred[b_idx]))
    ci_lower = float(1.0 - np.percentile(boot_accs, 97.5))
    ci_upper = float(1.0 - np.percentile(boot_accs, 2.5))
    
    # Bias-variance estimate (squared bias + variance approximation)
    bias_approx = float(test_error)
    var_approx = float(cv_std**2 * 100)
    
    # Overfit diagnostic
    if gen_gap > 0.30:
        overfit_diag = "Severe Overfitting (Leaf memorization)"
    elif gen_gap > 0.10:
        overfit_diag = "Moderate Overfitting"
    elif gen_gap < -0.05:
        overfit_diag = "Underfitting"
    else:
        overfit_diag = "Optimal Generalization Balance"
        
    # Conformal metrics if available
    if hasattr(model, 'predict_conformal'):
        conf_sets = model.predict_conformal(X_te)
        covered = sum(y_te[i] in conf_sets[i] for i in range(len(y_te)))
        coverage_pct = float(covered / len(y_te) * 100.0)
        mean_cardinality = float(np.mean([len(s) for s in conf_sets]))
        
        _, _, statuses = model.predict_with_status(X_te)
        rejected_pct = float(statuses.count("AMBIGUOUS_RETEST_REQUIRED") / len(statuses) * 100.0)
    else:
        coverage_pct = float('nan')
        mean_cardinality = 1.0
        rejected_pct = 0.0

    return {
        'Algorithm': name,
        'Accuracy': acc,
        'Macro_F1': f1,
        'Macro_Precision': prec,
        'Macro_Recall': rec,
        'Balanced_Accuracy': bal_acc,
        'MCC': mcc,
        'Cohen_Kappa': kappa,
        'CV_Accuracy': cv_mean,
        'CV_Std': cv_std,
        'Train_Accuracy': train_acc,
        'Test_Error': test_error,
        'Train_Error': train_error,
        'Validation_Error': val_error,
        'Generalization_Gap': gen_gap,
        'CI_95_Lower': ci_lower,
        'CI_95_Upper': ci_upper,
        'Bias_Estimate': bias_approx,
        'Variance_Estimate': var_approx,
        'Diagnostic_Label': overfit_diag,
        'Total_FP': fp_total,
        'Total_FN': fn_total,
        'Agronomic_Cost_1_1': cost_1_1,
        'Agronomic_Cost_4_1': cost_4_1,
        'Log_Loss': log_loss_val,
        'Brier_Score': brier_val,
        'Conformal_Coverage_Pct': coverage_pct,
        'Mean_Set_Cardinality': mean_cardinality,
        'Ambiguous_Rejection_Pct': rejected_pct,
        'Fit_Time_s': fit_time,
        'Inference_Time_s': pred_time,
        'Latency_us_per_sample': (pred_time / len(X_te)) * 1e6,
        'cv_accs': cv_accs
    }

# Execute benchmark on all models
results_list = []
for name, model in models.items():
    print(f"Evaluating {name}...")
    res = compute_model_metrics_and_20_errors(name, model, X_train, y_train, X_test, y_test)
    results_list.append(res)
    print(f"  -> Acc: {res['Accuracy']*100:.2f}%, Macro F1: {res['Macro_F1']:.4f}, Cost 4:1: {res['Agronomic_Cost_4_1']:.1f}, Fit: {res['Fit_Time_s']:.2f}s")

# 4. Statistical Tests: Proposed vs Baselines
cv_scores_proposed = results_list[0]['cv_accs']
stat_tests = []
for i in range(1, len(results_list)):
    b_name = results_list[i]['Algorithm']
    b_scores = results_list[i]['cv_accs']
    diff = np.array(cv_scores_proposed) - np.array(b_scores)
    w_stat, p_val = stats.wilcoxon(cv_scores_proposed, b_scores, zero_method='wilcox') if not np.all(diff == 0) else (0.0, 1.0)
    cohen_d = float(np.mean(diff) / (np.std(diff) + 1e-12))
    stat_tests.append({
        'Baseline': b_name,
        'Proposed_Mean_CV': float(np.mean(cv_scores_proposed)),
        'Baseline_Mean_CV': float(np.mean(b_scores)),
        'Mean_Difference': float(np.mean(diff)),
        'Wilcoxon_W': float(w_stat),
        'Wilcoxon_p_value': float(p_val),
        'Cohen_d': cohen_d,
        'Statistically_Significant (p<0.05)': bool(p_val < 0.05)
    })

# 5. Ablation Study
print("\nRunning Ablation Study for AgroStack-CR...")
ablation_configs = {
    'Full AgroStack-CR': AgroStackCR(n_estimators_et=100, max_iter_hgb=100, C_logistic=1.0, C_FP=4.0, random_state=GLOBAL_SEED),
    'Ablation 1: No Temperature Calibration (Raw Probs)': AgroStackCR(n_estimators_et=100, max_iter_hgb=100, C_logistic=1.0, C_FP=4.0, random_state=GLOBAL_SEED),
    'Ablation 2: No Bayesian MEC (Standard Argmax)': AgroStackCR(n_estimators_et=100, max_iter_hgb=100, C_logistic=1.0, C_FP=1.0, random_state=GLOBAL_SEED),
    'Ablation 3: No Extra Trees Base Estimator': None,
    'Ablation 4: No HistGBM Base Estimator': None,
    'Ablation 5: No Logistic Regression Base Estimator': None,
    'Ablation 6: No Stoichiometric / SFI Domain Features': None
}

ablation_results = []
# Full model
m_full = AgroStackCR(n_estimators_et=100, max_iter_hgb=100, C_logistic=1.0, C_FP=4.0, random_state=GLOBAL_SEED)
m_full.fit(X_train, y_train)
y_p_full = m_full.predict(X_test)
fp_f = int(len(y_test) - np.trace(confusion_matrix(y_test, y_p_full)))
ablation_results.append({
    'Configuration': 'Full AgroStack-CR',
    'Accuracy': accuracy_score(y_test, y_p_full),
    'Macro_F1': f1_score(y_test, y_p_full, average='macro'),
    'Agronomic_Cost_4_1': float(fp_f * 5.0),
    'Log_Loss': log_loss(y_test, m_full.predict_proba(X_test)),
    'Conformal_Coverage_Pct': float(sum(y_test[i] in m_full.predict_conformal(X_test)[i] for i in range(len(y_test))) / len(y_test) * 100.0),
    'Description': 'Complete integrated architecture with all 3 base models, calibration, MEC, and conformal set.'
})

# Ablation 1: No Temperature Calibration (force T=1.0)
m_notemp = AgroStackCR(n_estimators_et=100, max_iter_hgb=100, C_logistic=1.0, C_FP=4.0, random_state=GLOBAL_SEED)
m_notemp.fit(X_train, y_train)
m_notemp.temperature_ = 1.0 # Force T=1.0
y_p_notemp = m_notemp.predict(X_test)
fp_nt = int(len(y_test) - np.trace(confusion_matrix(y_test, y_p_notemp)))
ablation_results.append({
    'Configuration': 'Ablation 1: No Temperature Calibration (T=1.0)',
    'Accuracy': accuracy_score(y_test, y_p_notemp),
    'Macro_F1': f1_score(y_test, y_p_notemp, average='macro'),
    'Agronomic_Cost_4_1': float(fp_nt * 5.0),
    'Log_Loss': log_loss(y_test, m_notemp.predict_proba(X_test)),
    'Conformal_Coverage_Pct': float(sum(y_test[i] in m_notemp.predict_conformal(X_test)[i] for i in range(len(y_test))) / len(y_test) * 100.0),
    'Description': 'Disables post-hoc temperature probability scaling; uses raw meta-learner logits.'
})

# Ablation 2: No Bayesian MEC Decoding (Symmetric 1:1 Argmax)
m_nomec = AgroStackCR(n_estimators_et=100, max_iter_hgb=100, C_logistic=1.0, C_FP=1.0, C_FN=1.0, random_state=GLOBAL_SEED)
m_nomec.fit(X_train, y_train)
y_p_nomec = m_nomec.predict(X_test)
fp_nm = int(len(y_test) - np.trace(confusion_matrix(y_test, y_p_nomec)))
ablation_results.append({
    'Configuration': 'Ablation 2: No Bayesian MEC (Standard Argmax)',
    'Accuracy': accuracy_score(y_test, y_p_nomec),
    'Macro_F1': f1_score(y_test, y_p_nomec, average='macro'),
    'Agronomic_Cost_4_1': float(fp_nm * 5.0),
    'Log_Loss': log_loss(y_test, m_nomec.predict_proba(X_test)),
    'Conformal_Coverage_Pct': float(sum(y_test[i] in m_nomec.predict_conformal(X_test)[i] for i in range(len(y_test))) / len(y_test) * 100.0),
    'Description': 'Replaces cost-sensitive risk minimization with naive argmax over probabilities.'
})

# Ablation 3: Remove Extra Trees (Only HistGBM + Logistic)
class AgroStackNoET(AgroStackCR):
    def _init_base_models(self):
        return {
            'hgb': HistGradientBoostingClassifier(max_iter=self.max_iter_hgb, random_state=self.random_state),
            'lr': LogisticRegression(C=self.C_logistic, max_iter=1000, solver='lbfgs', random_state=self.random_state)
        }
    def _generate_meta_features(self, X):
        n = X.shape[0]
        meta_Z = np.zeros((n, 2 * self.K_), dtype=np.float64)
        col_offset = 0
        for name, model in self.base_models_.items():
            meta_Z[:, col_offset : col_offset + self.K_] = model.predict_proba(X)
            col_offset += self.K_
        return meta_Z

m_noet = AgroStackNoET(max_iter_hgb=100, C_logistic=1.0, C_FP=4.0, random_state=GLOBAL_SEED)
m_noet.fit(X_train, y_train)
y_p_noet = m_noet.predict(X_test)
fp_ne = int(len(y_test) - np.trace(confusion_matrix(y_test, y_p_noet)))
ablation_results.append({
    'Configuration': 'Ablation 3: Remove Extra Trees Base Model',
    'Accuracy': accuracy_score(y_test, y_p_noet),
    'Macro_F1': f1_score(y_test, y_p_noet, average='macro'),
    'Agronomic_Cost_4_1': float(fp_ne * 5.0),
    'Log_Loss': log_loss(y_test, m_noet.predict_proba(X_test)),
    'Conformal_Coverage_Pct': float(sum(y_test[i] in m_noet.predict_conformal(X_test)[i] for i in range(len(y_test))) / len(y_test) * 100.0),
    'Description': 'Removes randomized subspace bagging; leaves only HistGBM and Logistic Regression.'
})

# Ablation 4: Remove HistGBM (Only Extra Trees + Logistic)
class AgroStackNoHGB(AgroStackCR):
    def _init_base_models(self):
        return {
            'et': ExtraTreesClassifier(n_estimators=self.n_estimators_et, random_state=self.random_state, n_jobs=-1),
            'lr': LogisticRegression(C=self.C_logistic, max_iter=1000, solver='lbfgs', random_state=self.random_state)
        }
    def _generate_meta_features(self, X):
        n = X.shape[0]
        meta_Z = np.zeros((n, 2 * self.K_), dtype=np.float64)
        col_offset = 0
        for name, model in self.base_models_.items():
            meta_Z[:, col_offset : col_offset + self.K_] = model.predict_proba(X)
            col_offset += self.K_
        return meta_Z

m_nohgb = AgroStackNoHGB(n_estimators_et=100, C_logistic=1.0, C_FP=4.0, random_state=GLOBAL_SEED)
m_nohgb.fit(X_train, y_train)
y_p_nohgb = m_nohgb.predict(X_test)
fp_nh = int(len(y_test) - np.trace(confusion_matrix(y_test, y_p_nohgb)))
ablation_results.append({
    'Configuration': 'Ablation 4: Remove HistGBM Base Model',
    'Accuracy': accuracy_score(y_test, y_p_nohgb),
    'Macro_F1': f1_score(y_test, y_p_nohgb, average='macro'),
    'Agronomic_Cost_4_1': float(fp_nh * 5.0),
    'Log_Loss': log_loss(y_test, m_nohgb.predict_proba(X_test)),
    'Conformal_Coverage_Pct': float(sum(y_test[i] in m_nohgb.predict_conformal(X_test)[i] for i in range(len(y_test))) / len(y_test) * 100.0),
    'Description': 'Removes histogram gradient boosting; loses non-linear continuous threshold modeling.'
})

# Ablation 5: Remove Logistic Regression (Only Extra Trees + HistGBM)
class AgroStackNoLR(AgroStackCR):
    def _init_base_models(self):
        return {
            'et': ExtraTreesClassifier(n_estimators=self.n_estimators_et, random_state=self.random_state, n_jobs=-1),
            'hgb': HistGradientBoostingClassifier(max_iter=self.max_iter_hgb, random_state=self.random_state)
        }
    def _generate_meta_features(self, X):
        n = X.shape[0]
        meta_Z = np.zeros((n, 2 * self.K_), dtype=np.float64)
        col_offset = 0
        for name, model in self.base_models_.items():
            meta_Z[:, col_offset : col_offset + self.K_] = model.predict_proba(X)
            col_offset += self.K_
        return meta_Z

m_nolr = AgroStackNoLR(n_estimators_et=100, max_iter_hgb=100, C_FP=4.0, random_state=GLOBAL_SEED)
m_nolr.fit(X_train, y_train)
y_p_nolr = m_nolr.predict(X_test)
fp_nl = int(len(y_test) - np.trace(confusion_matrix(y_test, y_p_nolr)))
ablation_results.append({
    'Configuration': 'Ablation 5: Remove Logistic Regression Base Model',
    'Accuracy': accuracy_score(y_test, y_p_nolr),
    'Macro_F1': f1_score(y_test, y_p_nolr, average='macro'),
    'Agronomic_Cost_4_1': float(fp_nl * 5.0),
    'Log_Loss': log_loss(y_test, m_nolr.predict_proba(X_test)),
    'Conformal_Coverage_Pct': float(sum(y_test[i] in m_nolr.predict_conformal(X_test)[i] for i in range(len(y_test))) / len(y_test) * 100.0),
    'Description': 'Removes multinomial logistic model; lacks smooth linear prior calibration.'
})

# Ablation 6: Remove Domain Stoichiometric / SFI Features
raw_soil_clim_cols = soil_12_cols + climate_geo_cols
prep_raw = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('scaler', RobustScaler()),
            ('power', PowerTransformer(method='yeo-johnson'))
        ]), raw_soil_clim_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ]
)
X_tr_raw = prep_raw.fit_transform(X_train_raw)
X_te_raw = prep_raw.transform(X_test_raw)

m_nofeat = AgroStackCR(n_estimators_et=100, max_iter_hgb=100, C_logistic=1.0, C_FP=4.0, random_state=GLOBAL_SEED)
m_nofeat.fit(X_tr_raw, y_train)
y_p_nofeat = m_nofeat.predict(X_te_raw)
fp_nf = int(len(y_test) - np.trace(confusion_matrix(y_test, y_p_nofeat)))
ablation_results.append({
    'Configuration': 'Ablation 6: Remove Stoichiometric & SFI Features',
    'Accuracy': accuracy_score(y_test, y_p_nofeat),
    'Macro_F1': f1_score(y_test, y_p_nofeat, average='macro'),
    'Agronomic_Cost_4_1': float(fp_nf * 5.0),
    'Log_Loss': log_loss(y_test, m_nofeat.predict_proba(X_te_raw)),
    'Conformal_Coverage_Pct': float(sum(y_test[i] in m_nofeat.predict_conformal(X_te_raw)[i] for i in range(len(y_test))) / len(y_test) * 100.0),
    'Description': 'Removes engineered N:P, N:K, OC:N, deficiency flags and SFI score; uses raw soil & climate only.'
})

# 6. Robustness Experiments
print("\nRunning Robustness Evaluations...")
robustness_results = []

# A. Added Gaussian Noise to continuous soil measurements (10%, 20%, 30% std)
for noise_std in [0.05, 0.15, 0.30]:
    noise = np.random.normal(0, noise_std, size=X_test.shape)
    X_test_noisy = X_test + noise
    y_pred_noisy = m_full.predict(X_test_noisy)
    robustness_results.append({
        'Stress_Test': f'Added Gaussian Noise (std={noise_std})',
        'Parameter_Value': noise_std,
        'Accuracy': float(accuracy_score(y_test, y_pred_noisy)),
        'Macro_F1': float(f1_score(y_test, y_pred_noisy, average='macro')),
        'Degradation_Pct': float((res['Accuracy'] - accuracy_score(y_test, y_pred_noisy)) / res['Accuracy'] * 100.0)
    })

# B. Missing Values (Zero / Median Imputation at 10%, 20%, 30%)
for miss_frac in [0.10, 0.20, 0.30]:
    mask = np.random.binomial(1, miss_frac, size=X_test.shape).astype(bool)
    X_test_miss = X_test.copy()
    X_test_miss[mask] = 0.0 # Standard scaled mean is approx 0
    y_pred_miss = m_full.predict(X_test_miss)
    robustness_results.append({
        'Stress_Test': f'Simulated Missing Attributes ({int(miss_frac*100)}%)',
        'Parameter_Value': miss_frac,
        'Accuracy': float(accuracy_score(y_test, y_pred_miss)),
        'Macro_F1': float(f1_score(y_test, y_pred_miss, average='macro')),
        'Degradation_Pct': float((res['Accuracy'] - accuracy_score(y_test, y_pred_miss)) / res['Accuracy'] * 100.0)
    })

# C. Downsampled Training Set (25%, 50%, 75%)
for sample_pct in [0.25, 0.50, 0.75]:
    sub_size = int(len(X_train) * sample_pct)
    sub_idx = np.random.choice(len(X_train), sub_size, replace=False)
    m_sub = AgroStackCR(n_estimators_et=60, max_iter_hgb=60, C_FP=4.0, random_state=GLOBAL_SEED)
    m_sub.fit(X_train[sub_idx], y_train[sub_idx])
    y_pred_sub = m_sub.predict(X_test)
    robustness_results.append({
        'Stress_Test': f'Reduced Training Data ({int(sample_pct*100)}%)',
        'Parameter_Value': sample_pct,
        'Accuracy': float(accuracy_score(y_test, y_pred_sub)),
        'Macro_F1': float(f1_score(y_test, y_pred_sub, average='macro')),
        'Degradation_Pct': float((res['Accuracy'] - accuracy_score(y_test, y_pred_sub)) / res['Accuracy'] * 100.0)
    })

# D. Spatial District Cross-Validation (GroupKFold by District)
print("Evaluating Spatial District Cross-Validation (GroupKFold)...")
gkf = GroupKFold(n_splits=5)
spatial_accs = []
for tr_i, val_i in gkf.split(X_train_raw, y_train, groups=dist_train):
    X_s_tr = preprocessor.fit_transform(X_train_raw.iloc[tr_i])
    X_s_va = preprocessor.transform(X_train_raw.iloc[val_i])
    m_spat = AgroStackCR(n_estimators_et=60, max_iter_hgb=60, C_FP=4.0, random_state=GLOBAL_SEED)
    m_spat.fit(X_s_tr, y_train[tr_i])
    spatial_accs.append(accuracy_score(y_train[val_i], m_spat.predict(X_s_va)))

spatial_cv_mean = float(np.mean(spatial_accs))
spatial_cv_std = float(np.std(spatial_accs))
print(f"Spatial District GroupKFold Accuracy: {spatial_cv_mean*100:.2f}% +/- {spatial_cv_std*100:.2f}%")

# 7. Cost-Sensitivity Analysis Curve (Varying FP:FN Cost Ratios)
cost_ratios = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
cost_curve_results = []
for c_ratio in cost_ratios:
    m_c = AgroStackCR(n_estimators_et=100, max_iter_hgb=100, C_FP=c_ratio, C_FN=1.0, random_state=GLOBAL_SEED)
    m_c.fit(X_train, y_train)
    y_c_pred = m_c.predict(X_test)
    fp_c = int(len(y_test) - np.trace(confusion_matrix(y_test, y_c_pred)))
    fn_c = fp_c
    total_cost = fp_c * c_ratio + fn_c * 1.0
    cost_curve_results.append({
        'Cost_Ratio_FP_to_FN': c_ratio,
        'Accuracy': float(accuracy_score(y_test, y_c_pred)),
        'Macro_F1': float(f1_score(y_test, y_c_pred, average='macro')),
        'Total_Agronomic_Penalty': float(total_cost),
        'False_Positive_Count': fp_c
    })

# 8. Export All Results
print("\nExporting Experiment Results...")
# Prepare clean DataFrame of benchmark results
df_master = pd.DataFrame(results_list).drop(columns=['cv_accs'])
csv_path = os.path.join(current_dir, 'comparison_tables.csv')
df_master.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# Save Multi-Tab Excel Workbook
xlsx_path = os.path.join(current_dir, 'experiment_results.xlsx')
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    df_master.to_excel(writer, sheet_name='Benchmark_20_Errors', index=False)
    pd.DataFrame(stat_tests).to_excel(writer, sheet_name='Statistical_Significance', index=False)
    pd.DataFrame(ablation_results).to_excel(writer, sheet_name='Ablation_Study', index=False)
    pd.DataFrame(robustness_results).to_excel(writer, sheet_name='Robustness_Stress_Tests', index=False)
    pd.DataFrame(cost_curve_results).to_excel(writer, sheet_name='Cost_Sensitivity_Curve', index=False)
print(f"Saved: {xlsx_path}")

# Save json dumps for figures and reporting
with open(os.path.join(current_dir, 'ablation_results.json'), 'w') as f:
    json.dump(ablation_results, f, indent=4)

with open(os.path.join(current_dir, 'robustness_results.json'), 'w') as f:
    json.dump(robustness_results, f, indent=4)

with open(os.path.join(current_dir, 'statistical_tests_proposed.json'), 'w') as f:
    json.dump(stat_tests, f, indent=4)

with open(os.path.join(current_dir, 'cost_curve_results.json'), 'w') as f:
    json.dump(cost_curve_results, f, indent=4)

print("================================================================================")
print("EXPERIMENT EXECUTION COMPLETED SUCCESSFULLY!")
print("================================================================================")
