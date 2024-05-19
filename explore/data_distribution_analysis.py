# data_distribution_analysis.py
import scipy.stats as stats
import logging

def data_distribution_analysis(data, name):
    """Perform normality tests and log the results."""
    for column in ['num1', 'num2', 'num3', 'num4', 'num5', 'numA']:
        stat, p = stats.shapiro(data[column].astype(int))
        logging.info(f"Normality test for {column} in {name} dataset: Statistics={stat}, p={p}")

# No main function, this script will be called from explore_main.py