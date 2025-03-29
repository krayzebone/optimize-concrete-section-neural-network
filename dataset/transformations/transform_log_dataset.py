import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- Step 0: Read your original dataset ---
df_original = pd.read_parquet(r"C:\Users\marci\Desktop\Nowy SGU\dataset\files\dataset.parquet")

# Make a copy so we don't overwrite the original DataFrame
df_transformed = df_original.copy()

# --- Step 1: Log1p-transform skewed columns ---
skewed_cols = ['MRd', 'b', 'd', 'Mcr', 'wk', 'ro1', 'ro2']
for col in skewed_cols:
    # Safely apply log(1 + x)
    df_transformed[col] = np.log1p(df_transformed[col])

# --- Step 2: Encode discrete/categorical features ---
categorical_cols = ['a1', 'fi', 'fck']
for col in categorical_cols:
    df_transformed[col] = LabelEncoder().fit_transform(
        df_transformed[col].astype(str)
    )

# --- Step 3: Scale all numeric columns ---
numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])

# --- Step 4: Save or inspect the transformed DataFrame ---
df_transformed.to_parquet(r"C:\Users\marci\Desktop\Nowy SGU\dataset\files\dataset_transformed.parquet", index=False)

# 'df_transformed' now has all your transformations applied.
print("Transformation complete. New dataset saved to 'transformed_data.csv'.")
