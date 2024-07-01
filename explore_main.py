# explore_main.py

import pandas as pd
import logging
import os
from explore.categorical_analysis import CategoricalAnalysis
from explore.data_distribution_analysis import DataDistributionAnalysis
from explore.data_visualization import DataVisualization
from explore.descriptive_stats import DescriptiveStats
from explore.missing_values_analysis import MissingValuesAnalysis
from explore.time_series_analysis import TimeSeriesAnalysis
from explore.feature_engineering import FeatureEngineering

def ensure_directories_exist(dirs):
    """Ensure that the necessary directories exist."""
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def load_data(file_path):
    """Load the CSV data file."""
    if not os.path.exists(file_path):
        logging.error(f"Data file {file_path} not found.")
        return None
    return pd.read_csv(file_path)

def process_dataset(data, name):
    """Process a single dataset."""
    # Convert infinite values to NaN
    with pd.option_context('mode.use_inf_as_na', True):
        # Categorical analysis
        ca = CategoricalAnalysis(data, name)
        ca.analyze()

        # Data distribution analysis
        dda = DataDistributionAnalysis(data, name)
        dda.analyze()

        # Data visualization
        dv = DataVisualization(data, name)
        dv.visualize()

        # Descriptive statistics
        ds = DescriptiveStats(data, name)
        ds.analyze()

        # Missing values analysis
        mva = MissingValuesAnalysis(data, name)
        mva.analyze()

        # Time series analysis
        tsa = TimeSeriesAnalysis(data, name)
        tsa.analyze()

        # Feature engineering
        fe = FeatureEngineering(data, name)
        engineered_data = fe.engineer_features()

    return engineered_data

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the exploratory data analysis process...")

    # Define file paths and directories
    data_combined_path = 'data/data_combined.csv'
    data_pb_path = 'data/data_pb.csv'
    data_mb_path = 'data/data_mb.csv'
    output_dirs = ['explore/plots/combined', 'explore/plots/pb', 'explore/plots/mb', 'explore/engineered_data']
    
    ensure_directories_exist(output_dirs)

    # Load the datasets
    data_combined = load_data(data_combined_path)
    data_pb = load_data(data_pb_path)
    data_mb = load_data(data_mb_path)

    if data_combined is None or data_pb is None or data_mb is None:
        logging.error("Failed to load one or more datasets.")
        return

    # Process each dataset
    logging.info("Processing combined dataset...")
    data_combined = process_dataset(data_combined, 'combined')

    logging.info("Processing pb dataset...")
    data_pb = process_dataset(data_pb, 'pb')

    logging.info("Processing mb dataset...")
    data_mb = process_dataset(data_mb, 'mb')

    logging.info("Exploratory data analysis process completed.")

if __name__ == "__main__":
    main()