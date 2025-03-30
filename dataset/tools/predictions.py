# evaluation_script.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
import joblib
import matplotlib.pyplot as plt
from typing import Dict, Any

# ============================================
# Transformation Functions (same as in your training script)
# ============================================
def log_transform(x: np.ndarray) -> np.ndarray:
    """Natural log transformation with built-in epsilon handling."""
    return np.log(x + 1e-8)

def exp_inverse(x: np.ndarray) -> np.ndarray:
    """Exponential inverse transformation."""
    return np.exp(x)

# ============================================
# Configuration (aligned with your training script)
# ============================================
CONFIG: Dict[str, Any] = {
    'data': {
        'filepath': r"dataset/files/dataset_new.parquet",
        'features': ["b", "h", "d", "fi", "fck", "ro1", "ro2"],
        'target': "Mcr"
    },
    'paths': {
        'model': r"models\Mcr_model\model.keras",
        'scaler_X': r"models\Mcr_model\scaler_X.pkl",
        'scaler_y': r"models\Mcr_model\scaler_y.pkl",
        'transformers': r"models\Mcr_model\transformers_config.pkl"
    },
    'transformations': {
        'features': {
            'b': {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8},
            'h': {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8},
            'd': {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8},
            'fi': {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8},
            'fck': {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8},
            'ro1': {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8},
            'ro2': {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8},
        },
        'target': {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8}
    }
}

# ============================================
# Helper Functions
# ============================================
def load_and_prepare_data() -> pd.DataFrame:
    """Load and prepare the dataset."""
    if not os.path.exists(CONFIG['data']['filepath']):
        raise FileNotFoundError(f"File not found: {CONFIG['data']['filepath']}")
    
    df = pd.read_parquet(CONFIG['data']['filepath'])
    
    # Check for required columns
    required_columns = CONFIG['data']['features'] + [CONFIG['data']['target']]
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Dataset is missing required columns: {missing}")
    
    return df

def transform_features(X: np.ndarray) -> np.ndarray:
    """Apply feature transformations as defined in config."""
    X_transformed = np.zeros_like(X)
    for i, feature in enumerate(CONFIG['data']['features']):
        transform_config = CONFIG['transformations']['features'].get(
            feature, 
            {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8}
        )
        X_transformed[:, i] = transform_config['transform'](
            X[:, i] + transform_config['epsilon']
        )
    return X_transformed

def inverse_transform_target(y: np.ndarray) -> np.ndarray:
    """Inverse transform target values to original scale."""
    transform_config = CONFIG['transformations']['target']
    return transform_config['inverse_transform'](y)

# ============================================
# Main Evaluation Function
# ============================================
def evaluate_model_on_samples(n_samples: int = 50000, random_state: int = 42) -> None:
    """Evaluate the model on random samples from the dataset."""
    # Load data
    df = load_and_prepare_data()
    
    # Take random samples
    sample_df = df.sample(n_samples, random_state=random_state)
    X_sample = sample_df[CONFIG['data']['features']].values
    y_actual = sample_df[CONFIG['data']['target']].values
    
    # Check for non-positive values before transformation
    if np.any(X_sample <= 0):
        raise ValueError("All feature values must be positive for log transformation")
    
    # Load model and scalers
    if not all(os.path.exists(path) for path in CONFIG['paths'].values()):
        missing = [path for path in CONFIG['paths'].values() if not os.path.exists(path)]
        raise FileNotFoundError(f"Missing model files: {missing}")
    
    model = load_model(CONFIG['paths']['model'])
    scaler_X = joblib.load(CONFIG['paths']['scaler_X'])
    scaler_y = joblib.load(CONFIG['paths']['scaler_y'])
    
    # Transform features (same as during training)
    X_sample_transformed = transform_features(X_sample)
    
    # Scale features
    X_sample_scaled = scaler_X.transform(X_sample_transformed)
    
    # Make predictions
    predictions_scaled = model.predict(X_sample_scaled)
    
    # Inverse transform predictions
    predictions_transformed = scaler_y.inverse_transform(predictions_scaled)
    predictions = inverse_transform_target(predictions_transformed).flatten()
    
    # Calculate ratios
    ratios = predictions / y_actual
    
    # Print statistics
    def print_stats(name: str, ratio_array: np.ndarray) -> None:
        mean_ratio = np.mean(ratio_array)
        median_ratio = np.median(ratio_array)
        std_ratio = np.std(ratio_array)
        min_ratio = np.min(ratio_array)
        max_ratio = np.max(ratio_array)
        within_1_percent = np.sum((ratio_array >= 0.99) & (ratio_array <= 1.01)) / len(ratio_array) * 100

        print(f"=== {name} Ratio Stats ===")
        print(f" Mean   : {mean_ratio:.4f}")
        print(f" Median : {median_ratio:.4f}")
        print(f" Std    : {std_ratio:.4f}")
        print(f" Min    : {min_ratio:.4f}")
        print(f" Max    : {max_ratio:.4f}")
        print(f" % within 1% of 1.0: {within_1_percent:.2f}%")
        print("")
    
    print_stats("Predicted/Actual", ratios)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.scatter(y_actual, predictions, s=1, alpha=0.5)
    plt.plot([y_actual.min(), y_actual.max()], 
             [y_actual.min(), y_actual.max()], 
             'r--', label='Perfect prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Values (n={n_samples})')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot histogram of ratios
    plt.figure(figsize=(12, 6))
    plt.hist(ratios, bins=100, edgecolor='black')
    plt.axvline(1.0, color='red', linestyle='--', label='Ideal ratio (1.0)')
    plt.xlabel('Predicted / Actual Ratio')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Ratios')
    plt.grid(True)
    plt.legend()
    plt.show()

# ============================================
# Main Execution
# ============================================
if __name__ == "__main__":
    evaluate_model_on_samples(n_samples=50000, random_state=42)