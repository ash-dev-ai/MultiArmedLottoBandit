# reinforced_predictions.py


import os
import logging
import pandas as pd
import numpy as np
import glob
from reinforce.reinforce_rules import ReinforceRules  # Assuming the rules are in reinforce_rules.py

class PredictionAggregator:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.setup_logging()

    @staticmethod
    def setup_logging():
        """Setup the logging format and level."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname=s - %(message=s')

    @staticmethod
    def ensure_directories_exist(dirs):
        """Ensure that the necessary directories exist."""
        for directory in dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def load_predictions(self, dataset_name):
        """Load all prediction files for a given dataset from the specified directory."""
        pattern = f'{self.input_dir}/*_{dataset_name}_predictions_*.csv'
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

    def generate_multiple_predictions(self, predictions_list, num_predictions=5):
        """Generate multiple likely predictions from the list of predictions."""
        if not predictions_list:
            logging.error("No predictions to aggregate.")
            return None

        combined_predictions = pd.concat(predictions_list, axis=0)
        logging.info(f"Combined shape: {combined_predictions.shape}")

        # Generate multiple predictions
        diverse_predictions = []
        for i in range(num_predictions):
            sample_predictions = combined_predictions.sample(n=len(predictions_list), replace=True).mean(axis=0)
            diverse_predictions.append(sample_predictions)

        return diverse_predictions

    def map_predictions_to_numbers(self, predictions, number_type):
        """Map the predictions to valid lottery number ranges."""
        mapped_predictions = []
        for prediction in predictions:
            mapped = prediction.apply(lambda x: int(round(x)) % number_type['num1-num5'] + 1)
            mapped_predictions.append(mapped)
        return mapped_predictions

    def validate_predictions(self, predictions, historical_data, number_type):
        """Validate predictions based on given rules."""
        rules = ReinforceRules(number_type)
        valid_predictions = []

        for prediction in predictions:
            num1_to_num5 = prediction[:5].values
            numA = prediction['numA']

            prediction_dict = {
                'num1': num1_to_num5[0],
                'num2': num1_to_num5[1],
                'num3': num1_to_num5[2],
                'num4': num1_to_num5[3],
                'num5': num1_to_num5[4],
                'numA': numA,
                'numSum': num1_to_num5.sum(),
            }

            if rules.validate_all_rules(prediction_dict, historical_data):
                valid_predictions.append(prediction)

        return valid_predictions

    def make_final_predictions_for_dataset(self, dataset_name, number_type, output_file, num_predictions=5):
        """Generate and save final predictions for the specified dataset."""
        logging.info(f"Generating final predictions for {dataset_name}...")

        predictions_list = self.load_predictions(dataset_name)

        if not predictions_list:
            logging.error(f"No predictions found for {dataset_name}. Skipping final predictions.")
            return

        multiple_predictions = self.generate_multiple_predictions(predictions_list, num_predictions)

        if multiple_predictions is None:
            logging.error(f"Failed to generate multiple predictions for {dataset_name}.")
            return

        # Convert aggregated predictions to the expected number range
        columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA']
        final_predictions = []

        for prediction in multiple_predictions:
            prediction_df = pd.DataFrame([prediction], columns=columns)
            prediction_df['numA'] = int(round(prediction['numA'])) % number_type['numA'] + 1
            final_predictions.append(prediction_df)

        mapped_predictions = self.map_predictions_to_numbers(final_predictions, number_type)

        # Load historical data
        historical_data_path = f'data/data_{dataset_name}.csv'
        historical_data = pd.read_csv(historical_data_path)[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']]

        # Validate predictions
        validated_predictions = self.validate_predictions(mapped_predictions, historical_data, number_type)

        # Save all predictions to a single CSV file
        if validated_predictions:
            all_predictions_df = pd.concat(validated_predictions, axis=0)
            all_predictions_df.to_csv(f"{output_file}.csv", index=False, header=False)
            logging.info(f"Saved final predictions to {output_file}.csv")
            logging.info(f"Final predictions from {output_file}.csv:\n{all_predictions_df}")
        else:
            logging.error("No valid predictions to save.")

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime=s - %(levelname=s - %(message=s')
    input_dir = 'data/predictions'
    aggregator = PredictionAggregator(input_dir)

    number_types = {
        'combined': {'num1-num5': 70, 'numA': 26},
        'pb': {'num1-num5': 69, 'numA': 26},
        'mb': {'num1-num5': 70, 'numA': 25}
    }

    output_dir = 'reinforce/reinforcement_results'
    aggregator.ensure_directories_exist([output_dir])

    for dataset_name, number_type in number_types.items():
        output_file = os.path.join(output_dir, f"{dataset_name}_final_predictions")
        aggregator.make_final_predictions_for_dataset(dataset_name, number_type, output_file, num_predictions=5)
