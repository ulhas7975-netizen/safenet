import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DATA_DIR = r"C:\Users\HP\Desktop\safenet\datasets"  # where your raw CSVs are stored
CLEAN_DATA_PATH = "clean_balanced_dataset.csv"
FINAL_FEATURES_PATH = "final_features_dataset.csv"
TARGET_COL = "label"  # 🔸 change this to your dataset's target column

# ------------------------------------------------------------
# PHASE 1 + 2: CLEANING, ENCODING, NORMALIZATION, SMOTE
# ------------------------------------------------------------
def phase2_preprocessing():
    print("\n🔹 Phase 2: Cleaning and Balancing Data")

    # Step 1: Load and combine all CSV files
    csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("❌ No CSV files found in", DATA_DIR)
        return None

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
            print(f"Loaded: {f} → {df.shape}")
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")

    combined_df = pd.concat(dfs, ignore_index=True)
    print("✅ Combined shape:", combined_df.shape)

    # Step 2: Handle missing values
    combined_df = combined_df.replace(["?", "NA", "N/A", "null"], np.nan)
    combined_df = combined_df.dropna(axis=1, thresh=len(combined_df) * 0.7)  # drop cols with >30% missing
    combined_df.fillna(combined_df.median(numeric_only=True), inplace=True)

    # Step 3: Encode categorical features
    for col in combined_df.select_dtypes(include=["object"]).columns:
        combined_df[col] = LabelEncoder().fit_transform(combined_df[col].astype(str))

    # Step 4: Normalize numeric columns
    scaler = StandardScaler()
    num_cols = combined_df.select_dtypes(include=["int64", "float64"]).columns
    combined_df[num_cols] = scaler.fit_transform(combined_df[num_cols])

    # Step 5: Balance dataset using SMOTE
    if TARGET_COL not in combined_df.columns:
        print(f"❌ Target column '{TARGET_COL}' not found!")
        print("Available columns:", list(combined_df.columns))
        return None

    X = combined_df.drop(columns=[TARGET_COL])
    y = combined_df[TARGET_COL]
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    balanced_df = pd.concat([X_res, y_res], axis=1)

    balanced_df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"✅ Saved balanced data → {CLEAN_DATA_PATH}")
    return balanced_df

# ------------------------------------------------------------
# PHASE 3: FEATURE ENGINEERING (Correlation + PCA)
# ------------------------------------------------------------
def phase3_feature_engineering():
    print("\n🔹 Phase 3: Feature Engineering")

    # Step 1: Load balanced dataset
    if os.path.exists(CLEAN_DATA_PATH):
        df = pd.read_csv(CLEAN_DATA_PATH, low_memory=False)
        print("✅ Loaded:", CLEAN_DATA_PATH)
    else:
        print("❌ Missing file:", CLEAN_DATA_PATH)
        print("Running Phase 2 automatically...")
        df = phase2_preprocessing()
        if df is None:
            print("❌ Cannot continue without balanced dataset.")
            return

    # Step 2: Remove highly correlated features
    corr_matrix = df.corr()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
    df_reduced = df.drop(columns=to_drop)
    print(f"🔻 Dropped {len(to_drop)} correlated features")

    # Step 3: Apply PCA if high dimensional
    X = df_reduced.drop(columns=[TARGET_COL])
    y = df_reduced[TARGET_COL]

    if X.shape[1] > 50:
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X)
        df_final = pd.DataFrame(X_pca)
        df_final[TARGET_COL] = y.values
        print(f"✅ PCA applied: reduced to {X_pca.shape[1]} components")
    else:
        df_final = df_reduced

    df_final.to_csv(FINAL_FEATURES_PATH, index=False)
    print(f"✅ Saved final feature set → {FINAL_FEATURES_PATH}")

# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting SafeNet Full Data Pipeline...")

    # Run Phase 2 only if file doesn't exist
    if not os.path.exists(CLEAN_DATA_PATH):
        phase2_preprocessing()
    else:
        print(f"✅ Skipping Phase 2 (found {CLEAN_DATA_PATH})")

    # Always run Phase 3
    phase3_feature_engineering()

    print("\n🏁 Pipeline completed successfully!")
