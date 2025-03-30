import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import joblib
import matplotlib.pyplot as plt
from typing import Dict, Any, Callable

tf.random.set_seed(38)

# ============================================
# Define named transformation functions (pickle-friendly)
# ============================================
def identity_transform(x: np.ndarray) -> np.ndarray:
    """Identity transformation (no change)."""
    return x

def log_transform(x: np.ndarray) -> np.ndarray:
    """Natural log transformation with built-in epsilon handling."""
    return np.log(x + 1e-8)

def exp_inverse(x: np.ndarray) -> np.ndarray:
    """Exponential inverse transformation."""
    return np.exp(x)

def sqrt_transform(x: np.ndarray) -> np.ndarray:
    """Square root transformation."""
    return np.sqrt(x)

def square_inverse(x: np.ndarray) -> np.ndarray:
    """Square inverse transformation."""
    return x**2

# ============================================
# CENTRALIZED CONFIGURATION
# ============================================

# 1. Data Configuration
DATA_CONFIG: Dict[str, Any] = {
    'filepath': r"dataset\files\dataset_new.parquet",
    'features': ["b", "h", "d", "fi", "fck", "ro1", "ro2"],
    'target': "Mcr",
    'test_size': 0.3,
    'random_state': 42
}

# 2. Transformation Configuration
TRANSFORMATION_CONFIG: Dict[str, Any] = {
    'features': {
        # Feature-specific transformations (applied before scaling)
        'b':     {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8},
        'h':     {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8},
        'd':     {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8},
        'fi':    {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8},
        'fck':   {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8},
        'ro1':   {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8},
        'ro2':   {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8},
    },

    'target':    {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 1e-8}
}

# 3. Scaling Configuration
SCALER_CONFIG: Dict[str, Any] = {
    'X_scaler': StandardScaler(),  # or MinMaxScaler()
    'y_scaler': StandardScaler()   # or MinMaxScaler()
}

# 4. Model Architecture
MODEL_CONFIG: Dict[str, Any] = {
    'hidden_layers': [
        {'units': 289, 'activation': 'relu', 'dropout': 0.00017408447601856974},
        {'units': 84, 'activation': 'relu', 'dropout': 0.0073718573268978585}
    ],
    'output_activation': 'linear'
}

# 5. Training Configuration
TRAINING_CONFIG: Dict[str, Any] = {
    'optimizer': Adam(learning_rate=6.509132184030181e-05),
    'loss': 'mse',
    'metrics': ['mse', 'mae'],
    'batch_size': 148,
    'epochs': 3000,
    'callbacks': [
        EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=50, min_lr=1e-8),
    ]
}

# 6. Output Configuration
OUTPUT_CONFIG: Dict[str, Any] = {
    'save_path': r"models\Mcr_model",
    'visualization': {
        'max_samples': 100000,
        'histogram_bins': 100
    },
    'save_transformers': True
}

# ============================================
# Data Loading and Preprocessing
# ============================================
def load_and_preprocess_data() -> tuple:
    """Load and preprocess data with feature-specific transformations."""
    try:
        df = pd.read_parquet(DATA_CONFIG['filepath'])
        
        # Apply feature-specific transformations
        X_transformed = np.zeros_like(df[DATA_CONFIG['features']].values)
        for i, feature in enumerate(DATA_CONFIG['features']):
            transform_config = TRANSFORMATION_CONFIG['features'].get(
                feature, 
                {'transform': log_transform, 'inverse_transform': exp_inverse, 'epsilon': 0}
            )
            X_transformed[:, i] = transform_config['transform'](
                df[feature].values + transform_config['epsilon']
            )
        
        y = df[DATA_CONFIG['target']].values.reshape(-1, 1)
        
        # Apply target transformation
        y_transformed = TRANSFORMATION_CONFIG['target']['transform'](
            y + TRANSFORMATION_CONFIG['target']['epsilon']
        )

        # Scale features and transformed target
        X_scaled = SCALER_CONFIG['X_scaler'].fit_transform(X_transformed)
        y_scaled = SCALER_CONFIG['y_scaler'].fit_transform(y_transformed)

        # Train-validation split
        X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(
            X_scaled, y_scaled, 
            test_size=DATA_CONFIG['test_size'], 
            random_state=DATA_CONFIG['random_state']
        )
        
        return X_train, X_val, y_train_scaled, y_val_scaled, df, X_scaled
    
    except Exception as e:
        print(f"Error during data loading/preprocessing: {str(e)}")
        raise

# ============================================
# Transformation Helpers
# ============================================
def inverse_transform_features(X_scaled: np.ndarray) -> np.ndarray:
    """Inverse transform features from scaled to original space."""
    try:
        X_unscaled = SCALER_CONFIG['X_scaler'].inverse_transform(X_scaled)
        X_original = np.zeros_like(X_unscaled)
        for i, feature in enumerate(DATA_CONFIG['features']):
            transform_config = TRANSFORMATION_CONFIG['features'].get(
                feature, 
                {'inverse_transform': log_transform}
            )
            X_original[:, i] = transform_config['inverse_transform'](X_unscaled[:, i])
        return X_original
    except Exception as e:
        print(f"Error during feature inverse transformation: {str(e)}")
        raise

def inverse_transform_target(y_scaled: np.ndarray) -> np.ndarray:
    """Inverse transform target from scaled to original space."""
    try:
        y_transformed = SCALER_CONFIG['y_scaler'].inverse_transform(y_scaled)
        y_original = TRANSFORMATION_CONFIG['target']['inverse_transform'](y_transformed)
        return y_original
    except Exception as e:
        print(f"Error during target inverse transformation: {str(e)}")
        raise

# ============================================
# Model Building
# ============================================
def build_model(input_shape: int) -> Sequential:
    """Build and compile the neural network model."""
    try:
        model = Sequential()
        model.add(Input(shape=(input_shape,)))
        
        # Add hidden layers from config
        for layer in MODEL_CONFIG['hidden_layers']:
            model.add(Dense(layer['units'], activation=layer['activation']))
            if layer['dropout'] > 0:
                model.add(Dropout(layer['dropout']))
        
        # Output layer
        model.add(Dense(1, activation=MODEL_CONFIG['output_activation']))
        
        # Compile with training config
        model.compile(
            optimizer=TRAINING_CONFIG['optimizer'],
            loss=TRAINING_CONFIG['loss'],
            metrics=TRAINING_CONFIG['metrics']
        )
        
        model.summary()
        return model
    
    except Exception as e:
        print(f"Error during model building: {str(e)}")
        raise

# ============================================
# Training
# ============================================
def train_model(model: Sequential, X_train: np.ndarray, y_train_scaled: np.ndarray, 
               X_val: np.ndarray, y_val_scaled: np.ndarray) -> tf.keras.callbacks.History:
    """Train the model with configured settings."""
    try:
        history = model.fit(
            X_train, y_train_scaled,
            validation_data=(X_val, y_val_scaled),
            epochs=TRAINING_CONFIG['epochs'],
            batch_size=TRAINING_CONFIG['batch_size'],
            callbacks=TRAINING_CONFIG['callbacks'],
            verbose=1
        )
        return history
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

# ============================================
# Evaluation
# ============================================
def evaluate_model(model: Sequential, X_val: np.ndarray, y_val_scaled: np.ndarray) -> tuple:
    """Evaluate model performance and return predictions."""
    try:
        # Evaluation in transformed (scaled) space
        val_loss, val_mse_scaled, val_mae_scaled = model.evaluate(X_val, y_val_scaled, verbose=0)
        print(f"\nMetrics in transformed space:")
        print(f"  - {TRAINING_CONFIG['metrics'][0]}: {val_mse_scaled}")
        print(f"  - {TRAINING_CONFIG['metrics'][1]}: {val_mae_scaled}")

        # Convert predictions from scaled -> transformed -> original
        val_pred_scaled = model.predict(X_val, verbose=0)
        val_pred = inverse_transform_target(val_pred_scaled)

        # Convert ground truth back to real scale
        y_val_unscaled = inverse_transform_target(y_val_scaled)

        # Compute real-scale metrics
        mse_unscaled = np.mean((val_pred - y_val_unscaled) ** 2)
        mae_unscaled = np.mean(np.abs(val_pred - y_val_unscaled))
        r2 = 1 - np.sum((y_val_unscaled - val_pred) ** 2) / np.sum((y_val_unscaled - np.mean(y_val_unscaled)) ** 2)

        print("\nReal-Scale Metrics:")
        print("  - MSE:", mse_unscaled)
        print("  - MAE:", mae_unscaled)
        print("  - R²:", r2)

        return val_pred, y_val_unscaled
    
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        raise

# ============================================
# Visualization
# ============================================
def plot_histograms(predicted_all: np.ndarray, actual_all: np.ndarray) -> None:
    """Plot histograms of prediction errors."""
    try:
        squared_errors = (predicted_all.flatten() - actual_all.flatten()) ** 2
        abs_errors = np.abs(predicted_all.flatten() - actual_all.flatten())

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(squared_errors, bins=OUTPUT_CONFIG['visualization']['histogram_bins'], edgecolor='black')
        plt.title("Histogram of Squared Errors")
        plt.xlabel("Squared Error")
        plt.ylabel("Count")

        plt.subplot(1, 2, 2)
        plt.hist(abs_errors, bins=OUTPUT_CONFIG['visualization']['histogram_bins'], edgecolor='black')
        plt.title("Histogram of Absolute Errors")
        plt.xlabel("Absolute Error")
        plt.ylabel("Count")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error during histogram plotting: {str(e)}")

def plot_scatter(actual_all: np.ndarray, predicted_all: np.ndarray) -> None:
    """Scatter plot of predicted vs actual values."""
    try:
        n_samples = min(OUTPUT_CONFIG['visualization']['max_samples'], len(actual_all))
        indices = np.random.choice(range(len(actual_all)), size=n_samples, replace=False)

        actual = actual_all[indices]
        predicted = predicted_all[indices]

        plt.figure(figsize=(8, 8))
        plt.scatter(actual, predicted, s=1, alpha=0.5)
        # 1:1 line
        _min, _max = min(actual), max(actual)
        plt.plot([_min, _max], [_min, _max], 'r--')
        plt.title("Predicted vs. Actual Values")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Error during scatter plot generation: {str(e)}")

# ============================================
# Main Execution
# ============================================
def main() -> None:
    """Main execution function."""
    try:
        # 1) Load and preprocess data
        X_train, X_val, y_train_scaled, y_val_scaled, df, X_scaled = load_and_preprocess_data()

        # 2) Build model
        model = build_model(input_shape=X_train.shape[1])

        # 3) Train model
        history = train_model(model, X_train, y_train_scaled, X_val, y_val_scaled)

        # 4) Evaluate model
        val_pred, y_val_unscaled = evaluate_model(model, X_val, y_val_scaled)

        # Generate full predictions on all data
        full_pred_scaled = model.predict(X_scaled, verbose=0)
        full_pred = inverse_transform_target(full_pred_scaled)
        actual_all = df[DATA_CONFIG['target']].values

        # 5) Visualizations
        plot_histograms(full_pred, actual_all)
        plot_scatter(actual_all, full_pred)

        # 6) Save outputs
        os.makedirs(OUTPUT_CONFIG['save_path'], exist_ok=True)
        model.save(os.path.join(OUTPUT_CONFIG['save_path'], "model.keras"))
        joblib.dump(SCALER_CONFIG['X_scaler'], os.path.join(OUTPUT_CONFIG['save_path'], "scaler_X.pkl"))
        joblib.dump(SCALER_CONFIG['y_scaler'], os.path.join(OUTPUT_CONFIG['save_path'], "scaler_y.pkl"))
        
        if OUTPUT_CONFIG['save_transformers']:
            joblib.dump(TRANSFORMATION_CONFIG, os.path.join(OUTPUT_CONFIG['save_path'], "transformers_config.pkl"))

        print("\n✅ Model, scalers, and transformation parameters saved successfully.")

    except Exception as e:
        print(f"\n❌ Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()