# proximity_analysis.py

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

class ProximityAnalysis:
    def __init__(self, predictions_dir, data_combined, data_pb, data_mb, output_dir):
        self.predictions_dir = predictions_dir
        self.data_combined = data_combined
        self.data_pb = data_pb
        self.data_mb = data_mb
        self.output_dir = output_dir

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        logging.info(f"Initialized ProximityAnalysis with: {predictions_dir}, {data_combined}, {data_pb}, {data_mb}, {output_dir}")

    def load_predictions(self):
        """Load all predictions from the specified directory."""
        prediction_files = [os.path.join(self.predictions_dir, f) for f in os.listdir(self.predictions_dir) if f.endswith('.csv')]
        predictions = {}
        for file in prediction_files:
            try:
                predictions[os.path.basename(file)] = pd.read_csv(file)
                logging.info(f"Loaded predictions from {file}")
            except Exception as e:
                logging.error(f"Error loading {file}: {e}")
        return predictions

    def get_next_draw_date(self, prediction_date, actual_dates):
        """Find the next draw date in the actual data that is on or after the prediction date."""
        future_dates = actual_dates[actual_dates >= prediction_date]
        if not future_dates.empty:
            return future_dates.iloc[0]
        else:
            return None

    def calculate_proximity(self, actual, predicted):
        """Calculate the proximity score between actual and predicted values."""
        proximity_scores = np.abs(actual - predicted)
        return proximity_scores.mean()  # Average proximity score for the dataset

    def run_analysis(self):
        """Run the proximity analysis and save results."""
        predictions = self.load_predictions()

        # Load actual datasets with draw dates
        actual_combined = pd.read_csv(self.data_combined, parse_dates=['draw_date'])
        actual_pb = pd.read_csv(self.data_pb, parse_dates=['draw_date'])
        actual_mb = pd.read_csv(self.data_mb, parse_dates=['draw_date'])

        results = []

        for pred_file, pred_data in predictions.items():
            # Extract prediction date from the filename
            pred_date_str = pred_file.split('_')[-1].replace('.csv', '')
            try:
                pred_date = datetime.strptime(pred_date_str, '%Y-%m-%d')
            except ValueError:
                logging.warning(f"Unable to parse prediction date from {pred_file}, skipping.")
                continue

            # Determine which dataset to compare with based on the file name
            if 'combined' in pred_file:
                actual_data = actual_combined
            elif 'pb' in pred_file:
                actual_data = actual_pb
            elif 'mb' in pred_file:
                actual_data = actual_mb
            else:
                logging.warning(f"Unknown dataset type in file {pred_file}, skipping.")
                continue

            # Find the next draw date for comparison
            next_draw_date = self.get_next_draw_date(pred_date, actual_data['draw_date'])
            if next_draw_date is None:
                logging.warning(f"No future draw dates found for predictions made on {pred_date} in {pred_file}.")
                continue

            # Get the actual results for the next draw date
            actual_results = actual_data[actual_data['draw_date'] == next_draw_date]

            if actual_results.empty:
                logging.warning(f"No actual results found for draw date {next_draw_date} in {pred_file}.")
                continue

            # Ensure prediction data matches the shape and order of actual data
            pred_data = pred_data.iloc[:len(actual_results)]  # Truncate to the length of actual results for fair comparison

            # Calculating proximity for each set of predictions
            proximity_score = self.calculate_proximity(
                actual_results[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].to_numpy(),
                pred_data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].to_numpy()
            )

            results.append({
                'prediction_file': pred_file,
                'proximity_score': proximity_score,
                'prediction_date': pred_date.strftime('%Y-%m-%d'),
                'next_draw_date': next_draw_date.strftime('%Y-%m-%d')
            })

            logging.info(f"Proximity score for {pred_file}: {proximity_score}")

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        today = datetime.today().strftime('%Y-%m-%d')
        output_file = os.path.join(self.output_dir, f'proximity_analysis_results_{today}.csv')
        results_df.to_csv(output_file, index=False)
        logging.info(f"Proximity analysis results saved to {output_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Example usage
    predictions_dir = 'data/predictions'
    data_combined = 'data/data_combined.csv'
    data_pb = 'data/data_pb.csv'
    data_mb = 'data/data_mb.csv'
    output_dir = 'analysis/proximity'

    proximity_analysis = ProximityAnalysis(predictions_dir, data_combined, data_pb, data_mb, output_dir)
    proximity_analysis.run_analysis()
