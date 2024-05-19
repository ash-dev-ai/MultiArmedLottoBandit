# stats.py
import os
import pandas as pd
import numpy as np
import logging
from scipy import stats

# Ensure the log directory exists
log_dir = os.path.join('prep', 'log')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_stats(data, name):
    """Calculate statistics for each column in the data and save to a text file."""
    stats_file = os.path.join(log_dir, f'{name}_stats.txt')
    with open(stats_file, 'w') as f:
        for column in ['num1', 'num2', 'num3', 'num4', 'num5', 'numA']:
            f.write(f"Statistics for {column}:\n")
            f.write(f"Mean: {data[column].astype(int).mean()}\n")
            f.write(f"Median: {data[column].astype(int).median()}\n")
            mode_result = stats.mode(data[column].astype(int))
            mode_value = mode_result.mode[0] if isinstance(mode_result.mode, np.ndarray) else mode_result.mode
            mode_count = mode_result.count[0] if isinstance(mode_result.count, np.ndarray) else mode_result.count
            f.write(f"Mode: {mode_value} (count: {mode_count})\n")
            f.write(f"Standard Deviation: {data[column].astype(int).std()}\n")
            f.write(f"Variance: {data[column].astype(int).var()}\n")
            f.write(f"Minimum: {data[column].astype(int).min()}\n")
            f.write(f"Maximum: {data[column].astype(int).max()}\n")
            f.write("\n")
    logging.info(f"Statistics for {name} saved to {stats_file}")

if __name__ == "__main__":
    logging.error("This script should be called from prep_main.py")