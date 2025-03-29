import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the Parquet file
file_path = r"C:\Users\marci\Desktop\Nowy SGU\dataset\dataset.parquet"
df = pd.read_parquet(file_path)

# Specify the features to plot
features = ["M_Ed", "M_Rd", "b", "d", "h", 'a_1', "fi_gl", "f_ck", "n1", "n2", "wk", "M_cr"]

# Create a copy of only the features you want, then apply log transform
df_log = df[features].apply(np.log1p)  # log(x + 1)

# Set up the matplotlib figure
plt.figure(figsize=(15, 10))

# Plot histograms for each feature (log-transformed)
for i, feature in enumerate(features, 1):
    plt.subplot(3, 4, i)  # Adjust the grid size based on the number of features
    sns.histplot(df_log[feature], kde=False, bins=100)
    plt.title(f'Log-Transformed Histogram of {feature}')
    plt.xlabel(f'{feature} (log scale)')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
