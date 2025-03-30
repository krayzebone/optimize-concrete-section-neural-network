import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def no_transform(x):
    return x

def log_transform(x):
    return np.log(x + 1e-8)  # Added small epsilon to handle zeros

def log_inverse(x):
    return np.exp(x) - 1e-8  # Subtract epsilon to maintain consistency

def sqrt_transform(x):
    return np.sqrt(x)

def sqrt_inverse(x):
    return x**2

def reciprocal_transform(x):
    return 1.0 / (x + 1e-8)  # Added small epsilon to handle zeros

def reciprocal_inverse(x):
    return (1.0 / x) - 1e-8  # Subtract epsilon to maintain consistency

DATA_CONFIG = {
    'filepath': r"dataset\files\dataset_new.parquet",
    'features': ["b", "h", "d", "fi", "fck", "ro1", "ro2", "Mcr", "MRd", "wk"],
}

# Read the dataset
df = pd.read_parquet(DATA_CONFIG['filepath'])

# Transformation configuration
TRANSFORMATION_CONFIG = {
    'features': {
        'b': {'transform': log_transform, 'inverse_transform': log_inverse},
        'h': {'transform': log_transform, 'inverse_transform': log_inverse},
        'd': {'transform': no_transform, 'inverse_transform': no_transform},
        'fi': {'transform': log_transform, 'inverse_transform': log_inverse},
        'fck': {'transform': log_transform, 'inverse_transform': log_inverse},
        'ro1': {'transform': log_transform, 'inverse_transform': log_inverse},
        'ro2': {'transform': log_transform, 'inverse_transform': log_inverse},
        'Mcr': {'transform': log_transform, 'inverse_transform': log_inverse},
        'MRd': {'transform': log_transform, 'inverse_transform': log_inverse},
        'wk': {'transform': log_transform, 'inverse_transform': log_inverse},
    }
}

# Apply transformations to each feature
df_transformed = df.copy()
for feature, config in TRANSFORMATION_CONFIG['features'].items():
    transform_func = config['transform']
    df_transformed[feature] = transform_func(df[feature])

# Set up the matplotlib figure
plt.figure(figsize=(15, 10))

# Plot histograms for each transformed feature
for i, feature in enumerate(DATA_CONFIG['features'], start=1):
    plt.subplot(3, 5, i)
    sns.histplot(df_transformed[feature], kde=False, bins=100)
    plt.title(f'Transformed: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Optional: Show original vs transformed distributions side by side
plt.figure(figsize=(20, 15))
for i, feature in enumerate(DATA_CONFIG['features'], start=1):
    plt.subplot(4, 5, i)
    sns.histplot(df[feature], kde=False, bins=100, color='blue', alpha=0.5)
    plt.title(f'Original: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    
    plt.subplot(4, 5, i+10)
    sns.histplot(df_transformed[feature], kde=False, bins=100, color='red', alpha=0.5)
    plt.title(f'Transformed: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()