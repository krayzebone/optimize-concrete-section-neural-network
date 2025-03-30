import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error
import matplotlib.pyplot as plt

# Load your dataset (update the file path accordingly)
data = pd.read_parquet(r'dataset\files\dataset_new.parquet')

# Separate features and target (ensure 'target' is the correct column name)
X = data.drop('Mcr', axis=1)
y = data['Mcr']

# Log transform X and y (ensure all values are positive)
X_log = np.log(X)
y_log = np.log(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

# Initialize the MLPRegressor with explicit parameters (including 2 hidden layers for example)
model = MLPRegressor(
    hidden_layer_sizes=(100, 50),   # Two hidden layers: first with 100 neurons, second with 50
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=0.001,
    power_t=0.5,
    max_iter=500,
    shuffle=True,
    random_state=42,
    tol=1e-4,
    verbose=False,
    warm_start=False,
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=False,
    validation_fraction=0.1,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    n_iter_no_change=10,
    max_fun=15000
)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)
median_abs = median_absolute_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)
print("Explained Variance Score:", explained_var)
print("Median Absolute Error:", median_abs)

# Plotting residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Optionally, perform cross-validation for more robust estimates
cv_scores = cross_val_score(model, X_log, y_log, cv=5, scoring='r2')
print("Cross-validated R² scores:", cv_scores)
print("Average R² score from cross-validation:", np.mean(cv_scores))
