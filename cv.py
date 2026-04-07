import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import spearmanr
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

# Define target — single space after "?" to match CSV column name exactly
target_col = 'Did the patient return to the ED within 30 days of index visit? '

# Convert all columns to numeric where possible
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where target is missing
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)

print(f"\nDataset shape: {df.shape}")
print(f"\nTarget distribution (counts):")
print(df[target_col].value_counts())
print(f"\nTarget distribution (proportions):")
print(df[target_col].value_counts(normalize=True).round(4))

# ============================================================
# 2. MISSING DATA INFORMATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: MISSING DATA INFORMATION")
print("=" * 60)

missing     = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(2)
missing_df  = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df  = missing_df[missing_df['Missing Count'] > 0].sort_values(
                  'Missing %', ascending=False)

print(f"\nTotal columns with missing values: {len(missing_df)}")
print(f"Total missing cells: {df.isnull().sum().sum()}")
print(f"\nTop 20 columns with most missing data:")
print(missing_df.head(20).to_string())

# ============================================================
# 3. PREPARE FEATURES AND TARGET
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: FEATURE PREPARATION")
print("=" * 60)

X_full = df.drop(columns=[target_col])
y      = df[target_col]

for col in X_full.columns:
    X_full[col] = pd.to_numeric(X_full[col], errors='coerce')

# Fill missing values with ZERO
X_full_filled = X_full.fillna(0)

# Remove constant columns
non_constant_cols = X_full_filled.columns[X_full_filled.std() > 0]
X_full_filled     = X_full_filled[non_constant_cols]

print(f"\nMissing value fill strategy : ZERO (0)")
print(f"Feature matrix shape (after removing constant columns): {X_full_filled.shape}")
print(f"Target shape: {y.shape}")
print(f"Remaining missing values in X: {X_full_filled.isnull().sum().sum()}")

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def get_top_n_features(score_dict, n):
    """Return top-n feature names sorted by score descending."""
    sorted_feats = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    return [f[0] for f in sorted_feats[:n]]

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

# ---- Pure Feature Agglomeration ----
def fa_feature_selection(X, y, n):
    """
    Pure FA algorithm — no RF, no Spearman, no mean, no external scoring:
    1. Fit FeatureAgglomeration: groups features by inter-feature similarity
       using hierarchical clustering on feature correlations
    2. Each feature gets a cluster label from FA
    3. Score every feature by its variance (pure intrinsic property of the feature)
    4. Select top-n features globally across ALL clusters by variance score
       (not one per cluster — top-n anywhere across all clusters)
    """
    X_arr      = X.values
    feat_names = np.array(X.columns)

    # K clusters — use enough clusters so features are meaningfully grouped
    K = min(max(n * 10, 30), X_arr.shape[1])

    # Fit FA: clusters features based on inter-feature correlation structure
    fa     = FeatureAgglomeration(n_clusters=K)
    fa.fit(X_arr)

    labels = fa.labels_   # cluster label per feature, shape (n_features,)

    # Score: variance of each feature — pure intrinsic FA-compatible metric
    # No target y involved, no external algorithm
    feature_variances = X_arr.var(axis=0)   # shape (n_features,)

    # Build score dict: all features across all clusters ranked by variance
    score_dict = {}
    for feat_idx, feat_name in enumerate(feat_names):
        score_dict[feat_name] = feature_variances[feat_idx]

    # Return top-n globally across all clusters
    return get_top_n_features(score_dict, n)

# ---- HVGS: highest variance ----
def hvgs_feature_selection(X, y, n):
    scores = dict(X.var())
    return get_top_n_features(scores, n)

# ---- Spearman Correlation ----
def spearman_feature_selection(X, y, n):
    y_arr  = y.values
    scores = {}
    for col in X.columns:
        corr, _ = spearmanr(X[col].values, y_arr)
        scores[col] = abs(corr) if not np.isnan(corr) else 0.0
    return get_top_n_features(scores, n)

# ============================================================
# 5. CROSS-VALIDATION FUNCTION
# ============================================================

def run_cv(method_name, X_sel, y):
    """
    FA, HVGS, Spearman  -> RF for CV
    RF, XGB, LR         -> own algorithm for CV
    """
    if method_name == 'RF':
        model = RandomForestClassifier(random_state=42)
    elif method_name == 'XGB':
        model = XGBClassifier(random_state=42,
                              eval_metric='logloss',
                              use_label_encoder=False)
    elif method_name == 'LR':
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        # FA, HVGS, Spearman use RF for CV evaluation only
        model = RandomForestClassifier(random_state=42)

    scores = cross_val_score(model, X_sel, y,
                             cv=cv_splitter, scoring='accuracy')
    return scores.mean()

# ============================================================
# 6. FEATURE SELECTION - TOP 5 FROM FULL DATASET
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: FEATURE SELECTION - TOP 5 FROM FULL DATASET")
print("=" * 60)

selectors = {
    'RF':       rf_feature_selection,
    'XGB':      xgb_feature_selection,
    'LR':       lr_feature_selection,
    'FA':       fa_feature_selection,
    'HVGS':     hvgs_feature_selection,
    'Spearman': spearman_feature_selection,
}

results = {}

for name, fn in selectors.items():
    print(f"\n--- {name} ---")
    top5 = fn(X_full_filled, y, 5)
    print(f"Top 5 features: {top5}")

    cv5_acc = run_cv(name, X_full_filled[top5], y)
    print(f"CV Accuracy (top 5): {cv5_acc:.4f}")

    results[name] = {
        'top5':    top5,
        'cv5_acc': cv5_acc,
    }

# ============================================================
# 7. REMOVE HIGHEST FEATURE -> RE-SELECT TOP 4 (REDUCED SET)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: REDUCED DATASET - TOP 4 FEATURES")
print("=" * 60)

for name, fn in selectors.items():
    print(f"\n--- {name} ---")
    highest = results[name]['top5'][0]
    print(f"Removing highest feature: '{highest}'")

    X_reduced = X_full_filled.drop(columns=[highest])
    top4      = fn(X_reduced, y, 4)
    print(f"Top 4 features (reduced): {top4}")

    results[name]['highest_feature'] = highest
    results[name]['top4']            = top4

# ============================================================
# 8. SUMMARY TABLE
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: SUMMARY TABLE")
print("=" * 60)

rows = []
for name in selectors:
    top5_str = "; ".join(results[name]['top5'])
    top4_str = "; ".join(results[name]['top4'])
    cv5_acc  = round(results[name]['cv5_acc'], 4)
    row      = [name, cv5_acc, top5_str, top4_str]
    rows.append(row)

summary_df = pd.DataFrame(
    rows,
    columns=['Method', 'CV5 Accuracy', 'Top5 Features', 'Top4 Features']
)

print(summary_df.to_string(index=False))

summary_df.to_csv('result.csv', index=False)
print("\nSummary saved to result.csv")
