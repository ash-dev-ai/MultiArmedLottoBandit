# data_distribution_analysis.py

import scipy.stats as stats
import logging
import pandas as pd
import os

class DataDistributionAnalysis:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.results_dir = f'explore/results/{name}'
        self.ensure_dir(self.results_dir)

    def ensure_dir(self, directory):
        """Ensure that a directory exists."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def normality_test(self, column):
        """Perform Shapiro-Wilk normality test and return results."""
        stat, p = stats.shapiro(self.data[column].astype(int))
        return stat, p

    def analyze(self):
        """Perform normality tests on specified columns and save the results."""
        columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA']
        results = []

        for column in columns:
            stat, p = self.normality_test(column)
            result = {
                'column': column,
                'statistic': stat,
                'p_value': p
            }
            results.append(result)
            logging.info(f"Normality test for {column} in {self.name} dataset: Statistics={stat}, p={p}")

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.results_dir, 'normality_test_results.csv'), index=False)

# No main function, this script will be called from explore_main.py