# -----------------------------------------------
# 🧩 Data Preprocessing Pipeline for SafeNet
# -----------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

# 1️⃣ Load your combined dataset
df = pd.read_csv("combined_dataset.csv")

print("Original shape:", df.shape)

# ------------------------------------------------
# 2️⃣ Handle missing or corrupted data
# ------------------------------------------------

# Option 1: Drop columns/rows with too many NaNs
df = df.dropna(axis=1, thresh=len(df) * 0.5)  # drop columns with >50% missing
df = df.dropna(axis=0, thresh=df.shape[1] * 0.5)  # drop rows with >50% missing

# Option 2: Fill missing numeric and categorical values
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("After handling missing data:", df.shape)

# ------------------------------------------------
# 3️⃣ Encode categorical features
# ------------------------------------------------
categorical_cols = df.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print(f"Encoded {len(categorical_cols)} categorical columns.")

# ------------------------------------------------
# 4️⃣ Normalize numerical columns
# ------------------------------------------------
numerical_cols = df.select_dtypes(include=[np.number]).columns

# You can use either StandardScaler or MinMaxScaler:
scaler = StandardScaler()
# scaler = MinMaxScaler()

df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("✅ Normalization complete.")

# ------------------------------------------------
# 5️⃣ Handle imbalance using SMOTE
# ------------------------------------------------
# ⚠️ You need a target column — update the name below!
TARGET_COL = 'label'  # change this to your actual target column name

if TARGET_COL in df.columns:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("✅ SMOTE applied.")
    print("Before SMOTE:", y.value_counts().to_dict())
    print("After SMOTE:", pd.Series(y_resampled).value_counts().to_dict())

    # Recombine into one dataframe
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                              pd.DataFrame(y_resampled, columns=[TARGET_COL])],
                             axis=1)
    df_resampled.to_csv("clean_balanced_dataset.csv", index=False)
    print("💾 Saved as clean_balanced_dataset.csv")
else:
    print("⚠️ Please update TARGET_COL with your label/target column name.")
