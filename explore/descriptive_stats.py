# descriptive_stats.py
import logging
import pandas as pd
import os

class DescriptiveStats:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.summary_dir = f'explore/summaries'
        self.ensure_dir(self.summary_dir)

    def ensure_dir(self, directory):
        """Ensure that a directory exists."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def generate_stats(self):
        """Generate descriptive statistics for the dataset."""
        desc_stats = self.data.describe()
        logging.info(f"Descriptive Statistics for {self.name} dataset:\n{desc_stats}")
        return desc_stats

    def save_stats(self):
        """Save descriptive statistics to a CSV file."""
        desc_stats = self.generate_stats()
        desc_stats.to_csv(f'{self.summary_dir}/{self.name}_descriptive_stats.csv')
        logging.info(f"Descriptive statistics for {self.name} dataset saved to {self.summary_dir}/{self.name}_descriptive_stats.csv")

    def analyze(self):
        """Generate and save descriptive statistics."""
        self.save_stats()

# No main function, this script will be called from explore_main.py
