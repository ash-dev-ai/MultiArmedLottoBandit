# explore_main.py
import logging
import pandas as pd
from explore.descriptive_stats import descriptive_stats
from explore.data_visualization import data_visualization
from explore.categorical_analysis import categorical_analysis
from explore.time_series_analysis import time_series_analysis
from explore.missing_values_analysis import missing_values_analysis
from explore.feature_engineering import feature_engineering
from explore.data_distribution_analysis import data_distribution_analysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_dataset(data, name):
    logging.info(f"Analyzing {name} dataset")
    descriptive_stats(data, name)
    data_visualization(data, name)
    categorical_analysis(data, name)
    time_series_analysis(data, name)
    missing_values_analysis(data, name)
    feature_engineering(data, name)
    data_distribution_analysis(data, name)

def main():
    # Load the datasets
    data_combined = pd.read_csv('data/data_combined.csv')
    data_pb = pd.read_csv('data/data_pb.csv')
    data_mb = pd.read_csv('data/data_mb.csv')
    
    # Perform exploratory data analysis on each dataset
    analyze_dataset(data_combined, 'combined')
    analyze_dataset(data_pb, 'pb')
    analyze_dataset(data_mb, 'mb')

if __name__ == "__main__":
    main()

