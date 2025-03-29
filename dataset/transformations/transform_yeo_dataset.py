import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder

# --- Step 0: Read your original dataset ---
df_original = pd.read_parquet(r"C:\Users\marci\Desktop\Nowy SGU\dataset\files\dataset.parquet")

# Make a copy so we don't overwrite the original DataFrame
df_transformed = df_original.copy()

# --- Step 1: Yeo-Johnson transform skewed columns ---
skewed_cols = ['MRd', 'b', 'd', 'Mcr', 'wk', 'ro1', 'ro2']
yeojohnson = PowerTransformer(method='yeo-johnson', standardize=False)

# Apply Yeo-Johnson to each skewed column
for col in skewed_cols:
    # Reshape to 2D array as required by sklearn
    transformed_data = yeojohnson.fit_transform(df_transformed[[col]])
    df_transformed[col] = transformed_data.flatten()

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

# --- Step 4: Save the transformed DataFrame ---
df_transformed.to_parquet(r"C:\Users\marci\Desktop\Nowy SGU\dataset\files\dataset_transformed.parquet", index=False)

# Optional: Save the transformers for later use
import joblib
joblib.dump(yeojohnson, r"C:\Users\marci\Desktop\Nowy SGU\dataset\files\yeojohnson_transformer.pkl")
joblib.dump(scaler, r"C:\Users\marci\Desktop\Nowy SGU\dataset\files\standard_scaler.pkl")

print("Yeo-Johnson transformation complete. New dataset saved.")