# reinforced_predictions.py

import logging
import pandas as pd
import numpy as np
import os
import glob

def ensure_directories_exist(dirs):
    """Ensure that the necessary directories exist."""
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def load_predictions(directory, dataset_name):
    """Load all prediction files for a given dataset from the specified directory."""
    pattern = f'{directory}/*_{dataset_name}_predictions_*.csv'
    prediction_files = glob.glob(pattern)
    
    logging.info(f"Found {len(prediction_files)} prediction files for {dataset_name}.")
    predictions_list = []

    for file in prediction_files:
        try:
            data = pd.read_csv(file)
            predictions_list.append(data)
            logging.info(f"Loaded predictions from {file}.")
        except Exception as e:
            logging.error(f"Failed to load predictions from {file}: {e}")

    return predictions_list

def generate_multiple_predictions(predictions_list, num_predictions=5):
    """Generate multiple likely predictions from the list of predictions."""
    if not predictions_list:
        logging.error("No predictions to aggregate.")
        return None

    combined_predictions = pd.concat(predictions_list)
    logging.info(f"Combined shape: {combined_predictions.shape}")

    # Generate multiple predictions
    diverse_predictions = []
    for i in range(num_predictions):
        sample_predictions = combined_predictions.sample(frac=1, axis=0, replace=True).mean(axis=0)
        diverse_predictions.append(sample_predictions)

    return diverse_predictions

def map_predictions_to_numbers(predictions, number_type):
    """Map the predictions to valid lottery number ranges."""
    mapped_predictions = []
    for prediction in predictions:
        mapped = prediction.apply(lambda x: int(round(x)) % number_type['num1-num5'] + 1)
        mapped_predictions.append(mapped)
    return mapped_predictions

def make_final_predictions_for_dataset(dataset_name, number_type, output_file, num_predictions=5):
    """Generate and save final predictions for the specified dataset."""
    logging.info(f"Generating final predictions for {dataset_name}...")

    prediction_dir = 'data/predictions'
    predictions_list = load_predictions(prediction_dir, dataset_name)

    if not predictions_list:
        logging.error(f"No predictions found for {dataset_name}. Skipping final predictions.")
        return

    multiple_predictions = generate_multiple_predictions(predictions_list, num_predictions)

    if multiple_predictions is None:
        logging.error(f"Failed to generate multiple predictions for {dataset_name}.")
        return

    # Convert aggregated predictions to the expected number range
    columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA']
    final_predictions = []

    for i, prediction in enumerate(multiple_predictions):
        prediction_df = pd.DataFrame([prediction], columns=columns)
        prediction_df['numA'] = int(round(prediction['numA'])) % number_type['numA'] + 1
        final_predictions.append(prediction_df)

    mapped_predictions = map_predictions_to_numbers(final_predictions, number_type)

    # Save each set of predictions to a CSV file
    for i, mapped in enumerate(mapped_predictions):
        mapped.to_csv(f"{output_file}_prediction_{i+1}.csv", index=False)
        logging.info(f"Saved prediction {i+1} to {output_file}_prediction_{i+1}.csv")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the prediction aggregation process...")

    # Define the datasets and their number type constraints
    number_types = {
        'combined': {'num1-num5': 70, 'numA': 26},
        'pb': {'num1-num5': 69, 'numA': 26},
        'mb': {'num1-num5': 70, 'numA': 25}
    }

    output_dir = 'reinforce/reinforcement_results'
    ensure_directories_exist([output_dir])

    # Generate and save final predictions for each dataset
    for dataset_name, number_type in number_types.items():
        output_file = os.path.join(output_dir, f"{dataset_name}_final_predictions")
        make_final_predictions_for_dataset(dataset_name, number_type, output_file, num_predictions=5)

    logging.info("Prediction aggregation process completed.")

if __name__ == "__main__":
    main()
