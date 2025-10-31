# --------------------------------------------------------------
# 🚀 SAFENET: Data Cleaning + Feature Engineering Pipeline
# --------------------------------------------------------------
# Combines:
#   ✅ Phase 2: Data Cleaning & Preprocessing
#   ✅ Phase 3: Feature Engineering
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import chardet
import glob
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# --------------------------------------------------------------
# 1️⃣ Combine CSV files
# --------------------------------------------------------------
print("🔍 Loading and combining CSV files...")

csv_files = glob.glob(r"C:\Users\HP\Desktop\safenet\datasets\*.csv")

def read_file_with_detected_encoding(file):
    with open(file, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return pd.read_csv(file, encoding=result['encoding'], dtype=str)

combined_df = pd.concat(
    (read_file_with_detected_encoding(f) for f in csv_files),
    ignore_index=True
)

print(f"✅ Combined dataset shape: {combined_df.shape}")

# --------------------------------------------------------------
# 2️⃣ Handle missing or corrupted data
# --------------------------------------------------------------
print("\n🧹 Handling missing values...")

combined_df = combined_df.dropna(axis=1, thresh=len(combined_df) * 0.5)  # drop cols >50% missing
combined_df = combined_df.dropna(axis=0, thresh=combined_df.shape[1] * 0.5)  # drop rows >50% missing

for col in combined_df.columns:
    if combined_df[col].dtype in [np.float64, np.int64]:
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())
    else:
        combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])

print(f"✅ After missing data handling: {combined_df.shape}")

# --------------------------------------------------------------
# 3️⃣ Encode categorical features
# --------------------------------------------------------------
print("\n🔤 Encoding categorical columns...")

categorical_cols = combined_df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col].astype(str))
    label_encoders[col] = le

print(f"✅ Encoded {len(categorical_cols)} categorical features.")

# --------------------------------------------------------------
# 4️⃣ Normalize numerical features
# --------------------------------------------------------------
print("\n⚖️ Normalizing numeric features...")

numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
combined_df[numeric_cols] = scaler.fit_transform(combined_df[numeric_cols])

print("✅ Normalization complete.")

# --------------------------------------------------------------
# 5️⃣ Apply SMOTE for class imbalance
# --------------------------------------------------------------
TARGET_COL = 'label'  # ⚠️ Change to your actual target column name

if TARGET_COL not in combined_df.columns:
    raise ValueError(f"⚠️ Target column '{TARGET_COL}' not found. Please update this variable.")

print("\n📈 Applying SMOTE to balance dataset...")

X = combined_df.drop(columns=[TARGET_COL])
y = combined_df[TARGET_COL]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("✅ SMOTE applied successfully.")
print("Before SMOTE:", y.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_resampled).value_counts().to_dict())

balanced_df = pd.concat([
    pd.DataFrame(X_resampled, columns=X.columns),
    pd.DataFrame(y_resampled, columns=[TARGET_COL])
], axis=1)

balanced_df.to_csv("clean_balanced_dataset.csv", index=False)
print("💾 Saved clean_balanced_dataset.csv")

# --------------------------------------------------------------
# 6️⃣ Feature Engineering (Correlation + PCA)
# --------------------------------------------------------------
print("\n🔍 Performing feature engineering...")

X = balanced_df.drop(columns=[TARGET_COL])
y = balanced_df[TARGET_COL]

# Correlation-based feature reduction
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

print(f"🔻 Dropping {len(to_drop)} highly correlated features.")
X_reduced = X.drop(columns=to_drop)

# Optional: visualize correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap="coolwarm", cbar=False)
plt.title("Correlation Matrix (before reduction)")
plt.show()

# Apply PCA if dimensionality is high
if X_reduced.shape[1] > 30:
    print("\n⚙️ Applying PCA to reduce dimensionality...")
    X_scaled = scaler.fit_transform(X_reduced)
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    X_final = pd.DataFrame(X_pca)
    print(f"✅ PCA reduced features from {X_reduced.shape[1]} → {X_final.shape[1]}")
else:
    X_final = X_reduced

# Recombine and save
final_df = pd.concat([X_final, y.reset_index(drop=True)], axis=1)
final_df.to_csv("final_features_dataset.csv", index=False)
print("\n💾 Saved final_features_dataset.csv")
print(f"Final dataset shape: {final_df.shape}")

print("\n🎯 Phase 2 + Phase 3 completed successfully! Ready for model training.")
