# missing_values_analysis.py

import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import pandas as pd

class MissingValuesAnalysis:
    def __init__(self, data: pd.DataFrame, name: str):
        self.data = data
        self.name = name
        self.plot_dir = f'explore/plots/{name}'
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def visualize_missing_values(self):
        """Identify and visualize missing values."""
        plt.figure(figsize=(20, 20))
        sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
        plt.savefig(f'{self.plot_dir}/missing_values.png')
        plt.close()
        
    def get_missing_values_summary(self) -> pd.DataFrame:
        """Get a summary of missing values in the dataset."""
        missing_summary = self.data.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        missing_summary = missing_summary.to_frame().reset_index()
        missing_summary.columns = ['Column', 'Missing Values']
        return missing_summary
    
    def log_missing_values_summary(self):
        """Log the summary of missing values."""
        summary = self.get_missing_values_summary()
        if not summary.empty:
            logging.info(f"Missing values summary for {self.name} dataset:\n{summary}")
        else:
            logging.info(f"No missing values found in {self.name} dataset.")
    
    def save_missing_values_summary(self):
        """Save the summary of missing values to a CSV file."""
        summary = self.get_missing_values_summary()
        summary.to_csv(f'{self.plot_dir}/missing_values_summary.csv', index=False)
        
    def analyze(self):
        """Run all analysis methods."""
        self.visualize_missing_values()
        self.log_missing_values_summary()
        self.save_missing_values_summary()

# No main function, this script will be called from explore_main.py
