# descriptive_stats.py
import logging

def descriptive_stats(data, name):
    """Generate and log descriptive statistics for the dataset."""
    logging.info(f"Descriptive Statistics for {name} dataset:")
    logging.info(data.describe())

# No main function, this script will be called from explore_main.py



