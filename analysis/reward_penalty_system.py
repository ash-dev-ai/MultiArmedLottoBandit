# reward_penalty_system.py

import logging
import pandas as pd
import numpy as np
import os

class RewardPenaltySystem:
    def __init__(self, predictions_dir, data_dir, analysis_dir):
        """
        Initialize the RewardPenaltySystem with directories for predictions, data, and where to save analysis results.
        
        Parameters:
        - predictions_dir: Directory containing prediction CSV files.
        - data_dir: Directory containing actual data CSV files.
        - analysis_dir: Directory to save the analysis results.
        """
        self.predictions_dir = predictions_dir
        self.data_dir = data_dir
        self.analysis_dir = analysis_dir

        # Ensure the analysis directory exists
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)

    def run_reward_penalty_system(self):
        """Run the reward and penalty evaluation process."""
        logging.info("Running reward and penalty system...")

        # Load predictions
        predictions = self.load_predictions()

        # Load actuals
        actuals_combined = pd.read_csv(os.path.join(self.data_dir, 'data_combined.csv'))
        actuals_pb = pd.read_csv(os.path.join(self.data_dir, 'data_pb.csv'))
        actuals_mb = pd.read_csv(os.path.join(self.data_dir, 'data_mb.csv'))

        # Evaluate and assign rewards/penalties
        self.evaluate(predictions, actuals_combined, 'combined')
        self.evaluate(predictions, actuals_pb, 'pb')
        self.evaluate(predictions, actuals_mb, 'mb')

        logging.info("Reward and penalty system completed.")

    def load_predictions(self):
        """Load all predictions from the specified directory and categorize by dataset type."""
        predictions = {'combined': {}, 'pb': {}, 'mb': {}}
        
        files = [f for f in os.listdir(self.predictions_dir) if f.endswith('.csv')]
        if not files:
            logging.error(f"No prediction files found in {self.predictions_dir}.")
            return predictions
        
        for file in files:
            path = os.path.join(self.predictions_dir, file)
            try:
                model_name, dataset_type, _ = file.split('_')[:3]
                if dataset_type in predictions:
                    predictions[dataset_type][model_name] = pd.read_csv(path)
                    logging.info(f"Loaded predictions for {model_name} model ({dataset_type} dataset) from {file}")
            except Exception as e:
                logging.error(f"Error processing {file}: {e}")
        
        return predictions

    def evaluate(self, predictions, actuals, dataset_type):
        """Evaluate predictions against actuals for each model and calculate rewards/penalties."""
        logging.info(f"Evaluating reward and penalty system for {dataset_type} dataset...")
        
        for model, preds in predictions[dataset_type].items():
            score_summary = self.calculate_scores(preds, actuals, model, dataset_type)

            # Save the reward and penalty summary
            self.save_score_summary(model, dataset_type, score_summary)

    def calculate_scores(self, preds, actuals, model, dataset_type):
        """
        Calculate scores by comparing predictions with actual results.
        
        Parameters:
        - preds: DataFrame containing the predictions.
        - actuals: DataFrame containing the actual results.
        - model: Name of the prediction model.
        - dataset_type: Type of dataset being analyzed.
        
        Returns:
        A dictionary summarizing exact matches, partial matches, total points, and mean points per prediction.
        """
        logging.info(f"Calculating scores for {model} model ({dataset_type} dataset)...")

        # Initialize scores
        exact_matches = 0
        partial_matches = 0
        total_points = 0

        # Ensure predictions are rounded to nearest whole numbers
        preds = preds.round()

        for i, pred_row in preds.iterrows():
            if i >= len(actuals):
                logging.warning(f"Prediction row index {i} exceeds actuals length; skipping comparison.")
                continue

            actual_row = actuals.iloc[i]
            prediction = pred_row[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].values
            actual = actual_row[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].values

            # Calculate exact and partial matches
            exact_match = sum(prediction == actual)
            partial_match = len(set(prediction) & set(actual))

            exact_matches += exact_match
            partial_matches += partial_match

            # Assign points (example scoring system)
            points = exact_match * 10 + partial_match * 2
            total_points += points

        score_summary = {
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'total_points': total_points,
            'mean_points_per_prediction': total_points / len(preds) if len(preds) > 0 else 0
        }

        logging.info(f"Score Summary for {model} ({dataset_type}): {score_summary}")

        return score_summary

    def save_score_summary(self, model, dataset_type, summary):
        """
        Save the score summary to a CSV file.
        
        Parameters:
        - model: Name of the prediction model.
        - dataset_type: Type of dataset being analyzed.
        - summary: Dictionary summarizing the scores.
        """
        summary_file = os.path.join(self.analysis_dir, f'{model}_{dataset_type}_reward_penalty_summary.csv')
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"Score summary for {model} ({dataset_type}) saved to {summary_file}")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    predictions_dir = 'data/predictions'
    data_dir = 'data'
    analysis_dir = 'analysis/rewards_penalties'

    reward_penalty_system = RewardPenaltySystem(predictions_dir, data_dir, analysis_dir)
    reward_penalty_system.run_reward_penalty_system()

