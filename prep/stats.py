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


class DataStatistics:
    def __init__(self, data: pd.DataFrame, name: str):
        self.data = data
        self.name = name
        self.stats_file = os.path.join(log_dir, f'{name}_stats.txt')

    def calculate_row_stats(self):
        """Calculate statistics for each row in the data."""
        columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA']
        self.data['mean'] = self.data[columns].astype(int).mean(axis=1)
        self.data['median'] = self.data[columns].astype(int).median(axis=1)
        self.data['mode'] = self.data[columns].astype(int).mode(axis=1)[0]  # Mode might need special handling if there are multiple modes
        self.data['std_dev'] = self.data[columns].astype(int).std(axis=1)
        self.data['variance'] = self.data[columns].astype(int).var(axis=1)
        self.data['min'] = self.data[columns].astype(int).min(axis=1)
        self.data['max'] = self.data[columns].astype(int).max(axis=1)
        logging.info(f"Added row statistics columns to {self.name} data")
        return self.data

    def process(self):
        self.calculate_row_stats()
        self.save_stats_to_file()
        return self.data

    def save_stats_to_file(self):
        """Save statistics to a file (optional if you need to keep a record)"""
        with open(self.stats_file, 'w') as f:
            f.write(f"Statistics for {self.name}:\n")
            f.write(f"Columns: {self.data.columns.tolist()}\n")
            f.write(f"First few rows of data with stats:\n")
            f.write(self.data.head().to_string())
        logging.info(f"Statistics for {self.name} saved to {self.stats_file}")


def calculate_stats(data, name):
    stats_processor = DataStatistics(data, name)
    return stats_processor.process()
