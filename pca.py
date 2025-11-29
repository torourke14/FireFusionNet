import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------------------------------------
# 1. Select features (drop label + anything non-numeric)
# ------------------------------------------------------
label_col = "ignition"
feature_cols = [c for c in df.columns 
                if c != label_col and np.issubdtype(df[c].dtype, np.number)]

X = df[feature_cols].values

# Optional: subsample to speed things up
# Keep all positives, subsample negatives
ignition_mask = df[label_col] == 1
non_ignition_mask = ~ignition_mask

df_pos = df[ignition_mask]
df_neg = df[non_ignition_mask].sample(
    frac=0.05,  # e.g. 5% of non-ignite pixels
    random_state=42
)

df_sub = pd.concat([df_pos, df_neg], ignore_index=True)
X = df_sub[feature_cols].values

# ---------------------------------------------
# 2. Standardize features (critical for PCA)
# ---------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------
# 3. Fit PCA
# ---------------------------------------------
# You can either:
#   - Set n_components explicitly
#   - Or set None and look at cumulative variance
pca = PCA(n_components=None, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio per component
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print("Explained variance ratio:", explained_var)
print("Cumulative:", cumulative_var)

# We standardized all predictors and applied Principal Component Analysis (PCA). 
# The first k components explained ~X% of the total variance, capturing most of 
# the covariation structure among meteorological, land-cover, and anthropogenic 
# features.


### Mapping PCA components back to features (which features matter to PCs)
# pca.components_: shape (n_components, n_features)
# Each row is a component; each column corresponds to a feature in feature_cols

loadings = pd.DataFrame(
    pca.components_, 
    columns=feature_cols,
    index=[f"PC{i+1}" for i in range(pca.n_components_)]
)

num_pcs_to_inspect = 5

for pc in range(num_pcs_to_inspect):
    pc_name = f"PC{pc+1}"
    print(f"\nTop loadings for {pc_name}:")
    print(
        loadings.loc[pc_name]
        .abs()  # magnitude of contribution
        .sort_values(ascending=False)
        .head(10)
    )
    
#Interpretation:
# - High absolute loading = that feature heavily shapes that PC.
# - You can say things like:
#-  “PC1 represents a warm–dry gradient (high temperature, low RH, low soil moisture, 
# high vapor pressure deficit).”
# - “PC2 corresponds to topographic and land-cover variation (elevation, slope, forest cover).”


# You can also summarize overall PCA-based feature “importance” by aggregating 
# loadings across the first K PCs, weighted by explained variance:





from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.inspection import permutation_importance

# ------------------------------------------------------
# 1. Build X, y from the same subsampled df_sub
# ------------------------------------------------------
X = df_sub[feature_cols].values
y = df_sub[label_col].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Standardize for the RF? Not necessary, but for consistency:
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ------------------------------------------------------
# 2. Train a small but decent random forest
# ------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    class_weight="balanced_subsample",  # handle imbalance
    random_state=42
)

rf.fit(X_train_scaled, y_train)

# Base performance (AUPRC)
y_val_proba = rf.predict_proba(X_val_scaled)[:, 1]
base_auprc = average_precision_score(y_val, y_val_proba)
print("Baseline validation AUPRC:", base_auprc)

# ------------------------------------------------------
# 3. Permutation importance
# ------------------------------------------------------
def auprc_metric(estimator, X_val_scaled, y_val):
    proba = estimator.predict_proba(X_val_scaled)[:, 1]
    return average_precision_score(y_val, proba)

result = permutation_importance(
    rf,
    X_val_scaled,
    y_val,
    scoring=lambda est, Xb, yb: auprc_metric(est, Xb, yb),
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

rf_importances = pd.Series(
    result.importances_mean,
    index=feature_cols
).sort_values(ascending=False)

print("\nRandom forest permutation importances (AUPRC drop):")
print(rf_importances.head(20))