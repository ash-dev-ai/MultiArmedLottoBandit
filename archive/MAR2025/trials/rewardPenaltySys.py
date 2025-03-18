import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class RewardPenaltySystem:
    def __init__(self, predictions_file, data_file):
        """
        Initialize the RewardPenaltySystem with file paths for predictions and actual data.
        
        :param predictions_file: Path to the CSV file storing daily predictions with scores.
        :param data_file: Path to the CSV file containing the actual dataset used for model training.
        """
        self.predictions_file = predictions_file
        self.data_file = data_file

        # Load files or initialize empty DataFrame if predictions are not present
        self.predictions_df = self._load_csv(predictions_file)
        self.data_df = self._load_csv(data_file)

        # Check for the presence of the 'dataset' column
        self.has_dataset_column = 'dataset' in self.data_df.columns

        # Ensure 'date' and 'score' columns are present in predictions_df
        self._ensure_columns(['date', 'score'])

    def _load_csv(self, file_path):
        """
        Load a CSV file or return an empty DataFrame if the file doesn't exist.
        
        :param file_path: Path to the CSV file.
        :return: DataFrame of the loaded CSV data.
        """
        if os.path.exists(file_path):
            logging.info(f"Loaded data from {file_path}")
            return pd.read_csv(file_path)
        logging.warning(f"{file_path} not found. Initializing empty DataFrame.")
        return pd.DataFrame()

    def _ensure_columns(self, columns):
        """
        Ensure specified columns exist in predictions_df. Initialize them if missing.
        
        :param columns: List of column names to ensure in predictions_df.
        """
        for column in columns:
            if column not in self.predictions_df.columns:
                if column == 'score':
                    self.predictions_df[column] = 0  # Initialize missing scores to 0
                    logging.info(f"Initialized '{column}' column with default values.")
                else:
                    logging.error(f"Required column '{column}' is missing from predictions file.")
                    raise ValueError(f"Missing column: {column}")

    def update_scores(self):
        """
        Update the scores in the predictions file based on the latest predictions and actual data.
        """
        if self.predictions_df.empty or self.data_df.empty:
            logging.error("Predictions or actual data is empty. Cannot update scores.")
            return
        
        if 'date' not in self.predictions_df.columns or 'date' not in self.data_df.columns:
            logging.error("Date column is required in both predictions and actual data for scoring.")
            return

        # Loop through each prediction row and update scores individually
        for idx, prediction in self.predictions_df.iterrows():
            date = prediction['date']
            dataset_name = prediction.get('dataset', None)  # Handle case if 'dataset' column is missing
            rule_type = prediction['type']

            # Filter actual data based on date and dataset if 'dataset' column exists
            if self.has_dataset_column and dataset_name:
                actual_data = self.data_df[(self.data_df['date'] == date) & (self.data_df['dataset'] == dataset_name)]
            else:
                actual_data = self.data_df[self.data_df['date'] == date]

            # Handle if no matching actual data is found for this date and dataset
            if actual_data.empty:
                logging.info(f"No actual data found for date {date} and dataset {dataset_name}. Skipping score update.")
                continue

            # Calculate the score for this specific prediction
            score = self.calculate_score(prediction, actual_data.iloc[0])

            # Update the score in predictions_df for the current row only
            self.predictions_df.at[idx, 'score'] = score

        # Save updated predictions with scores
        self.predictions_df.to_csv(self.predictions_file, index=False)
        logging.info(f"Scores updated and saved to {self.predictions_file}")

    def calculate_score(self, prediction, actual):
        """
        Calculate the score for a single prediction based on actual results.
        
        :param prediction: Row containing the prediction data.
        :param actual: Row containing the actual results data.
        :return: Score as an integer or float.
        """
        score = 0
        # Scoring logic based on matching 'num1' to 'num5' and 'numA' columns
        for column in ['num1', 'num2', 'num3', 'num4', 'num5', 'numA']:
            if column in prediction and column in actual:
                if prediction[column] == actual[column]:
                    score += 10  # Full match
                elif abs(prediction[column] - actual[column]) <= 2:
                    score += 5  # Close match
                else:
                    score -= 1  # Penalty for mismatch
        return score

def main():
    logging.info("Starting reward-penalty scoring system.")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    predictions_file = os.path.join(root_dir, 'data', 'predictions.csv')
    data_file = os.path.join(root_dir, 'data', 'data_combined.csv')  # Replace with the relevant dataset file

    reward_system = RewardPenaltySystem(predictions_file, data_file)
    reward_system.update_scores()
    logging.info("Reward-penalty scoring process completed.")

if __name__ == "__main__":
    main()
