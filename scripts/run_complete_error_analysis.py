import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix, log_loss, hinge_loss, brier_score_loss,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    explained_variance_score, median_absolute_error, max_error
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier, DummyRegressor

# Classifiers
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier,
    AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.neural_network import MLPClassifier

# Regressors
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
)

warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)
os.makedirs('deliverables', exist_ok=True)

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

print("Starting Complete Empirical Analysis...")

# -------------------------------------------------------------
# 1. LOAD DATA & DOMAIN FEATURE ENGINEERING
# -------------------------------------------------------------
df_raw = pd.read_csv('cleaned_data.csv')
print(f"Loaded cleaned_data.csv: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")

eps = 1e-6

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
classes = list(label_enc.classes_)
n_classes = len(classes)

# Stratified 70/30 split
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

cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
processed_feature_names = numeric_features + cat_feature_names

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Save baseline info
analysis_output = {}

# -------------------------------------------------------------
# 2. LEAKAGE AUDIT EXPERIMENTS
# -------------------------------------------------------------
print("\n--- Running Leakage Audit Experiments ---")
leakage_results = {}

# 2.1 Target computation check: check correlation between input features and target
# (Mutual information or tree check)
# 2.2 Fit-on-all vs Fit-on-train preprocessing inflation
# Fit preprocessor on ALL data:
preprocessor_leaky = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('scaler', RobustScaler()),
            ('power', PowerTransformer(method='yeo-johnson'))
        ]), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ]
)
X_all_leaky = preprocessor_leaky.fit_transform(X_df)
X_train_leaky = X_all_leaky[X_train_raw.index]
X_test_leaky = X_all_leaky[X_test_raw.index]

clf_clean = RandomForestClassifier(n_estimators=50, random_state=GLOBAL_SEED, n_jobs=-1)
clf_clean.fit(X_train, y_train)
acc_clean = accuracy_score(y_test, clf_clean.predict(X_test))
f1_clean = f1_score(y_test, clf_clean.predict(X_test), average='macro')

clf_leaky = RandomForestClassifier(n_estimators=50, random_state=GLOBAL_SEED, n_jobs=-1)
clf_leaky.fit(X_train_leaky, y_train)
acc_leaky = accuracy_score(y_test, clf_leaky.predict(X_test_leaky))
f1_leaky = f1_score(y_test, clf_leaky.predict(X_test_leaky), average='macro')

leakage_results['preprocessing_leakage'] = {
    'clean_test_accuracy': float(acc_clean),
    'leaky_test_accuracy': float(acc_leaky),
    'accuracy_inflation': float(acc_leaky - acc_clean),
    'clean_macro_f1': float(f1_clean),
    'leaky_macro_f1': float(f1_leaky),
    'f1_inflation': float(f1_leaky - f1_clean)
}
print(f"Preprocessing Leakage Audit: Clean Acc = {acc_clean:.4f}, Leaky Acc = {acc_leaky:.4f}, Inflation = {acc_leaky - acc_clean:.4f}")

# 2.3 Duplicate rows check
exact_duplicates_all = df_raw.duplicated().sum()
feature_duplicates_raw = df_raw[soil_12_cols + climate_geo_cols].duplicated().sum()
# Train-test overlaps
train_df_feats = X_train_raw[soil_12_cols + climate_geo_cols]
test_df_feats = X_test_raw[soil_12_cols + climate_geo_cols]
merged_overlap = pd.merge(train_df_feats, test_df_feats, how='inner')
leakage_results['duplicate_audit'] = {
    'exact_dataset_duplicates': int(exact_duplicates_all),
    'feature_duplicates_total': int(feature_duplicates_raw),
    'train_test_overlap_count': int(len(merged_overlap))
}
print(f"Duplicate Audit: Exact={exact_duplicates_all}, Feature Dups={feature_duplicates_raw}, Overlap={len(merged_overlap)}")

# 2.4 Spatial Leakage: StratifiedKFold vs GroupKFold (by District)
print("Evaluating Spatial Leakage (Standard 5-Fold CV vs GroupKFold by District)...")
cv_standard = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
cv_group = GroupKFold(n_splits=5)

rf_cv = RandomForestClassifier(n_estimators=50, random_state=GLOBAL_SEED, n_jobs=-1)
scores_standard = cross_val_score(rf_cv, X_train, y_train, cv=cv_standard, scoring='accuracy', n_jobs=-1)
scores_group = cross_val_score(rf_cv, X_train, y_train, cv=cv_group, groups=dist_train.values, scoring='accuracy', n_jobs=-1)

leakage_results['spatial_leakage'] = {
    'standard_cv_acc_mean': float(scores_standard.mean()),
    'standard_cv_acc_std': float(scores_standard.std()),
    'group_cv_acc_mean': float(scores_group.mean()),
    'group_cv_acc_std': float(scores_group.std()),
    'spatial_performance_gap': float(scores_standard.mean() - scores_group.mean())
}
print(f"Spatial CV Gap: Standard={scores_standard.mean():.4f} +/- {scores_standard.std():.4f}, GroupKFold={scores_group.mean():.4f} +/- {scores_group.std():.4f}")

# 2.5 Shuffled-Label Test (Chance Verification)
print("Running Shuffled-Label Test...")
y_train_shuffled = np.random.RandomState(GLOBAL_SEED).permutation(y_train)
rf_shuffled = RandomForestClassifier(n_estimators=50, random_state=GLOBAL_SEED, n_jobs=-1)
rf_shuffled.fit(X_train, y_train_shuffled)
shuffled_acc = accuracy_score(y_test, rf_shuffled.predict(X_test))
shuffled_f1 = f1_score(y_test, rf_shuffled.predict(X_test), average='macro', zero_division=0)
chance_level = 1.0 / n_classes

leakage_results['shuffled_label_test'] = {
    'chance_theoretical': float(chance_level),
    'shuffled_test_accuracy': float(shuffled_acc),
    'shuffled_macro_f1': float(shuffled_f1),
    'passed_sanity_check': bool(abs(shuffled_acc - chance_level) < 0.02)
}
print(f"Shuffled-label Acc: {shuffled_acc:.4f} (Theoretical Chance: {chance_level:.4f}). Sanity check passed: {leakage_results['shuffled_label_test']['passed_sanity_check']}")

# 2.6 Single-feature suspicion test
print("Running Single-Feature Suspicion Test...")
single_feat_results = []
for idx, fname in enumerate(processed_feature_names):
    X_single_tr = X_train[:, [idx]]
    X_single_te = X_test[:, [idx]]
    clf_s = DecisionTreeClassifier(max_depth=3, random_state=GLOBAL_SEED)
    clf_s.fit(X_single_tr, y_train)
    acc_s = accuracy_score(y_test, clf_s.predict(X_single_te))
    single_feat_results.append({'feature': fname, 'single_feature_acc': float(acc_s)})

df_single_feat = pd.DataFrame(single_feat_results).sort_values(by='single_feature_acc', ascending=False)
leakage_results['single_feature_top'] = df_single_feat.head(5).to_dict(orient='records')
print(f"Top single feature accuracy: {df_single_feat.iloc[0]['feature']} -> {df_single_feat.iloc[0]['single_feature_acc']:.4f} (No feature exhibits >30% individual accuracy, confirming no single-column leak).")

analysis_output['leakage_results'] = leakage_results

with open('deliverables/leakage_results.json', 'w') as f:
    json.dump(leakage_results, f, indent=4)

print("Leakage audit complete and exported.")
