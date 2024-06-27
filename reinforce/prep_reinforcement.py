# prep_reinforcement.py
import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_sums(data, name):
    """Add numSum and totalSum columns to the data and log the operation."""
    data['numSum'] = data[['num1', 'num2', 'num3', 'num4', 'num5']].astype(int).sum(axis=1)
    data['totalSum'] = data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].astype(int).sum(axis=1)
    logging.info(f"Added numSum and totalSum columns to {name} data")
    return data

def preprocess_predictions(input_dir, output_dir):
    """Preprocess all prediction files in the input directory and save to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    for file in prediction_files:
        file_path = os.path.join(input_dir, file)
        data = pd.read_csv(file_path)
        data = add_sums(data, file)
        output_file_path = os.path.join(output_dir, file)
        data.to_csv(output_file_path, index=False)
        logging.info(f"Preprocessed and saved {file} to {output_dir}")

if __name__ == "__main__":
    input_dir = 'data/predictions'
    output_dir = 'data/preprocessed_predictions'
    preprocess_predictions(input_dir, output_dir)

