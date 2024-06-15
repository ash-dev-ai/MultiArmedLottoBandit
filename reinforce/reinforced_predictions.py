# reinforced_predictions.py

import logging
import pandas as pd
import numpy as np
import os
import glob

# Define the path to the predictions directory
PREDICTIONS_DIR = 'data/predictions'

def load_data(file_path):
    """Load the CSV data file."""
    if not os.path.exists(file_path):
        logging.error(f"Data file {file_path} not found.")
        return None
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess data to include only specified columns."""
    columns_to_use = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']
    return data[columns_to_use]

def map_actions_to_numbers(action, max_number):
    """Map an action to the lottery numbers based on the max_number."""
    return (action % max_number) + 1

def load_predictions_for_dataset(dataset_name):
    """Load and combine predictions for a specific dataset (combined, pb, mb) from various algorithms."""
    pattern = f'*_predictions_{dataset_name}_*.csv'
    prediction_files = glob.glob(os.path.join(PREDICTIONS_DIR, pattern))
    
    combined_predictions = []

    for file in prediction_files:
        if os.path.exists(file):
            df = pd.read_csv(file, index_col=0)
            combined_predictions.append(df)
        else:
            logging.warning(f"Prediction file {file} not found.")
    
    if combined_predictions:
        # Concatenate all the predictions into a single DataFrame along the columns
        combined_df = pd.concat(combined_predictions, axis=1)
        logging.info(f"Loaded predictions from {len(combined_predictions)} files for dataset: {dataset_name}.")
        return combined_df
    else:
        logging.error(f"No predictions loaded for dataset: {dataset_name}. Check the prediction file paths.")
        return None

def ensemble_predictions(predictions_df):
    """Ensemble predictions by averaging and returning as a DataFrame."""
    if predictions_df is not None:
        # Group by columns (there might be multiple predictions for the same 'numX' from different models)
        grouped = predictions_df.groupby(predictions_df.columns, axis=1)
        # Average predictions from each group
        ensemble_pred = grouped.mean()
        # Round to the nearest integer and clip to the valid range (assuming predictions should be whole numbers)
        ensemble_pred = ensemble_pred.round().clip(lower=1)
        return ensemble_pred
    else:
        logging.error("No predictions data provided for ensembling.")
        return None

def make_final_predictions_for_dataset(dataset_name, number_type, output_file):
    """Make final predictions for a specific dataset and save to the specified file."""
    logging.info(f"Making final predictions for {dataset_name} and saving to {output_file}...")

    # Load and combine predictions from the specified directory for this dataset
    combined_predictions = load_predictions_for_dataset(dataset_name)
    
    if combined_predictions is not None:
        # Perform ensemble prediction
        ensemble_pred = ensemble_predictions(combined_predictions)
        
        if ensemble_pred is not None:
            # Convert to the required format for final output
            final_predictions = []
            for _, row in ensemble_pred.iterrows():
                final_row = [map_actions_to_numbers(int(row[f'num{i+1}']), number_type['num1-num5']) for i in range(5)]
                final_row.append(map_actions_to_numbers(int(row['numA']), number_type['numA']))
                final_predictions.append(final_row)

            # Convert to DataFrame and save to CSV
            final_df = pd.DataFrame(final_predictions, columns=['num1', 'num2', 'num3', 'num4', 'num5', 'numA'])
            final_df.to_csv(output_file, index=False)
            logging.info(f"Final predictions saved to {output_file}")
        else:
            logging.error(f"Failed to ensemble predictions for dataset: {dataset_name}.")
    else:
        logging.error(f"Failed to load combined predictions for dataset: {dataset_name}. Skipping final prediction generation.")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the prediction process...")

    # Define the state and action space sizes
    state_space_size = 10  # Number of features used, including 'date'
    action_space_size = 70  # Max range to cover for num1-num5 for combined (1-70)

    # Define the datasets and their number type constraints
    datasets = {
        'combined': {'num1-num5': 70, 'numA': 26},
        'pb': {'num1-num5': 69, 'numA': 26},
        'mb': {'num1-num5': 70, 'numA': 25}
    }

    # Define paths for input data and output predictions
    base_path = 'data'
    output_dir = 'reinforce/reinforcement_results'
    ensure_directories_exist([output_dir])

    # Process each dataset
    for dataset_name, number_type in datasets.items():
        output_file = os.path.join(output_dir, f"{dataset_name}_final_predictions.csv")
        make_final_predictions_for_dataset(dataset_name, number_type, output_file)

    logging.info("Prediction process completed.")

def ensure_directories_exist(dirs):
    """Ensure that the necessary directories exist."""
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

if __name__ == "__main__":
    main()
