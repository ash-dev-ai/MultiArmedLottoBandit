# prep_reinforcement.py
import logging
import os
import pandas as pd

class Preprocessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.setup_logging()

    @staticmethod
    def setup_logging():
        """Setup the logging format and level."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    @staticmethod
    def add_sums(data: pd.DataFrame, name: str) -> pd.DataFrame:
        """Add numSum and totalSum columns to the data and log the operation."""
        if all(col in data.columns for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'numA']):
            data['numSum'] = data[['num1', 'num2', 'num3', 'num4', 'num5']].astype(int).sum(axis=1)
            data['totalSum'] = data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].astype(int).sum(axis=1)
            logging.info(f"Added numSum and totalSum columns to {name} data")
        elif 'numSum' in data.columns and 'totalSum' in data.columns:
            logging.info(f"File {name} already contains numSum and totalSum columns")
        else:
            logging.error(f"Missing required columns in {name}")
        return data

    def preprocess_predictions(self):
        """Preprocess all prediction files in the input directory and save to the output directory."""
        self.ensure_output_dir_exists()
        prediction_files = self.get_prediction_files()

        for file in prediction_files:
            file_path = os.path.join(self.input_dir, file)
            data = pd.read_csv(file_path)
            data = self.add_sums(data, file)
            output_file_path = os.path.join(self.output_dir, file)
            data.to_csv(output_file_path, index=False)
            logging.info(f"Preprocessed and saved {file} to {self.output_dir}")

    def ensure_output_dir_exists(self):
        """Ensure the output directory exists, create it if it doesn't."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_prediction_files(self) -> list:
        """Get a list of all prediction files in the input directory."""
        return [f for f in os.listdir(self.input_dir) if f.endswith('.csv')]

if __name__ == "__main__":
    input_dir = 'data/predictions'
    output_dir = 'data/preprocessed_predictions'
    preprocessor = Preprocessor(input_dir, output_dir)
    preprocessor.preprocess_predictions()
