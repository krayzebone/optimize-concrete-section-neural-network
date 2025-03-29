import pandas as pd
import numpy as np

# File paths
input_file = r"C:\Users\marci\Desktop\Nowy SGU\dataset.parquet"
output_file = r"C:\Users\marci\Desktop\Nowy SGU\dataset\files\dataset.parquet"
sample_size = 100000

# Read the original parquet file
try:
    df = pd.read_parquet(input_file)
    
    # Get the number of rows in the original file
    original_rows = len(df)
    print(f"Original dataset has {original_rows:,} rows")
    
    # Determine how many rows to sample
    n_samples = min(sample_size, original_rows)
    
    # Randomly sample rows without replacement
    sampled_df = df.sample(n=n_samples, replace=False, random_state=42)
    
    # Save the sampled data to a new parquet file
    sampled_df.to_parquet(output_file, index=False)
    print(f"Saved {n_samples:,} randomly sampled rows to {output_file}")
    
except FileNotFoundError:
    print(f"Error: File not found at {input_file}")
except Exception as e:
    print(f"An error occurred: {str(e)}")