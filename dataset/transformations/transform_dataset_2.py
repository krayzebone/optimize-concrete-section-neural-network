import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer

# Basic transformation functions (for non-power transforms)
def no_transform(x):
    return x

def log_transform(x):
    return np.log(x + 1e-8)

def log_inverse(x):
    return np.exp(x) - 1e-8

def sqrt_transform(x):
    return np.sqrt(x)

def sqrt_inverse(x):
    return x**2

def reciprocal_transform(x):
    return 1.0 / (x + 1e-8)

def reciprocal_inverse(x):
    return (1.0 / x) - 1e-8

DATA_CONFIG = {
    'filepath': r"dataset\files\dataset_new.parquet",
    'features': ["b", "h", "d", "fi", "fck", "ro1", "ro2", "Mcr", "MRd", "wk"],
}

# Read the dataset
df = pd.read_parquet(DATA_CONFIG['filepath'])

# Transformation configuration
TRANSFORMATION_CONFIG = {
    'features': {
        'b':    {'transform': 'log',    'inverse_transform': 'log_inverse'},
        'h':    {'transform': 'log',    'inverse_transform': 'log_inverse'},
        'd':    {'transform': 'log',    'inverse_transform': 'log_inverse'},
        'fi':   {'transform': 'log',    'inverse_transform': 'log_inverse'},
        'fck':  {'transform': 'log',    'inverse_transform': 'log_inverse'},
        'ro1':  {'transform': 'log',    'inverse_transform': 'log_inverse'},
        'ro2':  {'transform': 'log',    'inverse_transform': 'log_inverse'},
        'Mcr':  {'transform': 'log',    'inverse_transform': 'log_inverse'},
        'MRd':  {'transform': 'log',    'inverse_transform': 'log_inverse'},
        'wk':   {'transform': 'log',    'inverse_transform': 'log_inverse'},
    }
}

class FeatureTransformer:
    def __init__(self, config):
        self.config = config
        self.transformers = {}  # Stores PowerTransformer instances
    
    def transform_feature(self, x, feature_name):
        method = self.config['features'][feature_name]['transform']
        x = x.reshape(-1, 1)  # Reshape for sklearn
        
        if method in ['box-cox', 'yeo-johnson']:
            pt = PowerTransformer(method=method, standardize=False)
            transformed = pt.fit_transform(x)
            self.transformers[feature_name] = pt  # Store for inverse
            return transformed.flatten()
        elif method == 'log':
            return log_transform(x)
        elif method == 'sqrt':
            return sqrt_transform(x)
        elif method == 'reciprocal':
            return reciprocal_transform(x)
        else:  # no_transform
            return no_transform(x)
    
    def inverse_transform_feature(self, x, feature_name):
        method = self.config['features'][feature_name]['inverse_transform']
        x = x.reshape(-1, 1)  # Reshape for sklearn
        
        if method in ['box-cox', 'yeo-johnson']:
            return self.transformers[feature_name].inverse_transform(x).flatten()
        elif method == 'log_inverse':
            return log_inverse(x)
        elif method == 'sqrt_inverse':
            return sqrt_inverse(x)
        elif method == 'reciprocal_inverse':
            return reciprocal_inverse(x)
        else:  # no_transform
            return no_transform(x)

# Initialize transformer
transformer = FeatureTransformer(TRANSFORMATION_CONFIG)

# Apply transformations
df_transformed = df.copy()
for feature in DATA_CONFIG['features']:
    df_transformed[feature] = transformer.transform_feature(df[feature].values, feature)

# Plotting function
def plot_distributions(original_df, transformed_df, features):
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(features, start=1):
        # Original distribution
        plt.subplot(4, 5, i)
        sns.histplot(original_df[feature], kde=False, bins=100, color='blue', alpha=0.5)
        plt.title(f'Original: {feature}')
        plt.xlabel(feature)
        
        # Transformed distribution
        plt.subplot(4, 5, i+10)
        sns.histplot(transformed_df[feature], kde=False, bins=100, color='red', alpha=0.5)
        transform_method = TRANSFORMATION_CONFIG['features'][feature]['transform']
        plt.title(f'Transformed ({transform_method}): {feature}')
        plt.xlabel(feature)
    
    plt.tight_layout()
    plt.show()

# Plot original vs transformed distributions
plot_distributions(df, df_transformed, DATA_CONFIG['features'])

# Example of inverse transformation
sample_feature = 'b'
sample_data = df[sample_feature].values[:5]  # Take first 5 values
print("\nOriginal values:", sample_data)

transformed_data = transformer.transform_feature(sample_data, sample_feature)
print("Transformed values:", transformed_data)

restored_data = transformer.inverse_transform_feature(transformed_data, sample_feature)
print("Restored values:", restored_data)