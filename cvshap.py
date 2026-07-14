import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import spearmanr
import shap
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD AND PREPROCESS DATA
# ============================================================
df = pd.read_csv('EDSubstanceInducedPsychos_DATA.to share.csv',
                 na_values=['.', '', ' ', 'NA', 'NaN'])

print("=" * 60)
print("STEP 1: DATA LOADING AND PREPROCESSING")
print("=" * 60)

# Drop Record ID
df = df.drop(columns=['Record ID'], errors='ignore')

# Define target
target_col = 'Did the patient return to the ED within 30 days of index visit? '

# Convert all columns to numeric where possible
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where target is missing
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)

print(f"\nDataset shape after removing empty target: {df.shape}")
print(f"\nTarget distribution (counts):")
print(df[target_col].value_counts().sort_index())
print(f"\nTarget distribution (proportions):")
print(df[target_col].value_counts(normalize=True).sort_index().round(4))

# ============================================================
# 2. HANDLE MISSING DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: MISSING DATA HANDLING")
print("=" * 60)

# Separate features from target
X_full = df.drop(columns=[target_col])
y = df[target_col]

# Calculate missing percentage for each feature
missing_pct = (X_full.isnull().sum() / len(X_full)) * 100

print(f"\nOriginal number of features: {X_full.shape[1]}")

# Keep only columns with <= 50% missing
cols_to_keep = missing_pct[missing_pct <= 50].index.tolist()
X_full = X_full[cols_to_keep]

print(f"Features after removing >50% missing: {X_full.shape[1]}")
print(f"Features removed: {len(missing_pct) - len(cols_to_keep)}")

# Show missing data information before filling
missing_info = pd.DataFrame({
    'Missing Count': X_full.isnull().sum(),
    'Missing %': (X_full.isnull().sum() / len(X_full) * 100).round(2)
})
missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(
    'Missing %', ascending=False)

print(f"\nFeatures with missing data: {len(missing_info)}")
if len(missing_info) > 0:
    print(f"\nTop 10 features with most missing data:")
    print(missing_info.head(10).to_string())

# Fill missing values with zero
X_full_filled = X_full.fillna(0)

# Remove constant columns
non_constant_cols = X_full_filled.columns[X_full_filled.std() > 0]
X_full_filled = X_full_filled[non_constant_cols]

print(f"\nFeatures after removing constant columns: {X_full_filled.shape[1]}")
print(f"Missing values after filling with0: {X_full_filled.isnull().sum().sum()}")
print(f"\nFinal dataset shape: {X_full_filled.shape}")
print(f"Target shape: {y.shape}")

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random seed for SHAP sampling
SHAP_SAMPLE_SIZE = 100
SHAP_RANDOM_STATE = 42

def get_top_n_features(score_dict, n):
    """Return top-n feature names sorted by score descending."""
    sorted_feats = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    return [f[0] for f in sorted_feats[:n]]

def get_shap_sample(X, sample_size=SHAP_SAMPLE_SIZE, random_state=SHAP_RANDOM_STATE):
    """Randomly sample instances from X for SHAP computation."""
    if len(X) <= sample_size:
        return X
    return X.sample(n=sample_size, random_state=random_state)

# ---- RF Feature Selection ----
def rf_feature_selection(X, y, n):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    scores = dict(zip(X.columns, model.feature_importances_))
    return get_top_n_features(scores, n)

# ---- XGB Feature Selection ----
def xgb_feature_selection(X, y, n):
    model = XGBClassifier(random_state=42,
                          eval_metric='logloss',
                          use_label_encoder=False)
    model.fit(X, y)
    scores = dict(zip(X.columns, model.feature_importances_))
    return get_top_n_features(scores, n)

# ---- Logistic Regression Feature Selection ----
def lr_feature_selection(X, y, n):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    scores = dict(zip(X.columns, np.abs(model.coef_[0])))
    return get_top_n_features(scores, n)

# ---- RF-SHAP Feature Selection ----
def rf_shap_feature_selection(X, y, n):
    """Fit RF on full data, compute SHAP importance on a random 100-sample subset."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    X_sample = get_shap_sample(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    # shap_values shape: (n_samples, n_features) -> handle 3D case (n_samples, n_features, n_classes)
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    scores = dict(zip(X.columns, mean_abs_shap))
    return get_top_n_features(scores, n)

# ---- XGB-SHAP Feature Selection ----
def xgb_shap_feature_selection(X, y, n):
    """Fit XGB on full data, compute SHAP importance on a random 100-sample subset."""
    model = XGBClassifier(random_state=42,
                          eval_metric='logloss',
                          use_label_encoder=False)
    model.fit(X, y)

    X_sample = get_shap_sample(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    scores = dict(zip(X.columns, mean_abs_shap))
    return get_top_n_features(scores, n)

# ---- LR-SHAP Feature Selection ----
def lr_shap_feature_selection(X, y, n):
    """Fit LR on full data, compute SHAP importance on a random 100-sample subset."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)

    X_sample = get_shap_sample(X)

    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X_sample)

    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    scores = dict(zip(X.columns, mean_abs_shap))
    return get_top_n_features(scores, n)

# ---- Feature Agglomeration ----
def fa_feature_selection(X, y, n):
    """
    Feature Agglomeration: select top features across all clusters
    """
    X_arr = X.values
    feat_names = np.array(X.columns)
    
    # Number of clusters
    K = min(max(n * 10, 30), X_arr.shape[1])
    
    # Fit Feature Agglomeration
    fa = FeatureAgglomeration(n_clusters=K)
    fa.fit(X_arr)
    
    # Score each feature by its variance
    feature_variances = X_arr.var(axis=0)
    
    # Build score dict for all features
    score_dict = {}
    for feat_idx, feat_name in enumerate(feat_names):
        score_dict[feat_name] = feature_variances[feat_idx]
    
    # Return top-n globally across all clusters
    return get_top_n_features(score_dict, n)

# ---- HVGS: Highly Variable Gene Selection ----
def hvgs_feature_selection(X, y, n):
    """Select features with highest variance"""
    scores = dict(X.var())
    return get_top_n_features(scores, n)

# ---- Spearman Correlation ----
def spearman_feature_selection(X, y, n):
    """Select features with highest absolute Spearman correlation with target"""
    y_arr = y.values
    scores = {}
    for col in X.columns:
        corr, _ = spearmanr(X[col].values, y_arr)
        scores[col] = abs(corr) if not np.isnan(corr) else 0.0
    return get_top_n_features(scores, n)

# ============================================================
# 4. CROSS-VALIDATION FUNCTION
# ============================================================

def run_cv(method_name, X_sel, y):
    """
    FA, HVGS, Spearman -> RF for CV
    RF, XGB, LR, RF-SHAP, XGB-SHAP, LR-SHAP -> own base algorithm for CV
    """
    if method_name in ('RF', 'RF-SHAP'):
        model = RandomForestClassifier(random_state=42)
    elif method_name in ('XGB', 'XGB-SHAP'):
        model = XGBClassifier(random_state=42,
                              eval_metric='logloss',
                              use_label_encoder=False)
    elif method_name in ('LR', 'LR-SHAP'):
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        # FA, HVGS, Spearman use RF for CV evaluation
        model = RandomForestClassifier(random_state=42)
    
    scores = cross_val_score(model, X_sel, y,
                             cv=cv_splitter, scoring='accuracy')
    return scores.mean()

# ============================================================
# 5. FEATURE SELECTION - TOP 6 FROM FULL DATASET
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: FEATURE SELECTION - TOP 6 FROM FULL DATASET")
print("=" * 60)

selectors = {
    'RF': rf_feature_selection,
    'XGB': xgb_feature_selection,
    'LR': lr_feature_selection,
    'RF-SHAP': rf_shap_feature_selection,
    'XGB-SHAP': xgb_shap_feature_selection,
    'LR-SHAP': lr_shap_feature_selection,
    'FA': fa_feature_selection,
    'HVGS': hvgs_feature_selection,
    'Spearman': spearman_feature_selection,
}

results = {}

for name, fn in selectors.items():
    print(f"\n--- {name} ---")
    top6 = fn(X_full_filled, y, 6)
    print(f"Top 6 features: {top6}")
    
    cv6_acc = run_cv(name, X_full_filled[top6], y)
    print(f"CV Accuracy (top 6): {cv6_acc:.4f}")
    
    results[name] = {
        'top6': top6,
        'cv6_acc': cv6_acc,
    }

# ============================================================
# 6. REMOVE HIGHEST FEATURE -> RE-SELECT TOP 5 (REDUCED SET)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: REDUCED DATASET - TOP 5 FEATURES")
print("=" * 60)

for name, fn in selectors.items():
    print(f"\n--- {name} ---")
    highest = results[name]['top6'][0]
    print(f"Removing highest feature: '{highest}'")
    
    X_reduced = X_full_filled.drop(columns=[highest])
    top5 = fn(X_reduced, y, 5)
    print(f"Top 5 features (reduced): {top5}")
    
    results[name]['highest_feature'] = highest
    results[name]['top5'] = top5

# ============================================================
# 7. SUMMARY TABLE
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: SUMMARY TABLE")
print("=" * 60)

rows = []
for name in selectors:
    top6_str = "; ".join(results[name]['top6'])
    top5_str = "; ".join(results[name]['top5'])
    cv6_acc = round(results[name]['cv6_acc'], 4)
    row = [name, cv6_acc, top6_str, top5_str]
    rows.append(row)

summary_df = pd.DataFrame(
    rows,
    columns=['Method', 'CV6 Accuracy', 'Top6 Features', 'Top5 Features']
)

print(summary_df.to_string(index=False))

summary_df.to_csv('result.csv', index=False)
print("\n" + "=" * 60)
print("Summary saved to result.csv")
print("=" * 60)