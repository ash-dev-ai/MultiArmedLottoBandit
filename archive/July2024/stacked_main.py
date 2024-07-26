#stacked_main.py

import logging
import pandas as pd
from datetime import datetime
import os

# Import ensemble models
from stacked.rnn_ensemble import RNNEnsemble
from stacked.deep_learning_ensemble import DeepLearningEnsemble
from stacked.reservoir_ensemble import ReservoirEnsemble
from stacked.meta_learner import MetaLearner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_directories_exist(dirs):
    """Ensure that the necessary directories exist."""
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def load_and_preprocess_data(file_paths):
    """Load and preprocess data from the given file paths."""
    data = {}
    for dataset_type, path in file_paths.items():
        if not os.path.exists(path):
            logging.error(f"Data file {path} not found.")
            continue
        data[dataset_type] = pd.read_csv(path)
    return data

def save_predictions(predictions, dataset_type, model_name):
    """Save the predictions to a CSV file."""
    today = datetime.today().strftime('%Y-%m-%d')
    predictions_dir = 'data/predictions'
    ensure_directories_exist([predictions_dir])
    predictions_file = os.path.join(predictions_dir, f'{model_name}_{dataset_type}_predictions_{today}.csv')
    
    predictions_df = pd.DataFrame(predictions, columns=['num1', 'num2', 'num3', 'num4', 'num5', 'numA'])
    predictions_df['numSum'] = predictions_df[['num1', 'num2', 'num3', 'num4', 'num5']].sum(axis=1)
    predictions_df['totalSum'] = predictions_df['numSum'] + predictions_df['numA']
    
    predictions_df.to_csv(predictions_file, index=False)
    logging.info(f"Predictions saved to {predictions_file}")

def main():
    logging.info("Starting the stacked learning process...")

    # Define file paths
    file_paths = {
        'train_combined': 'data/train_combined.csv',
        'val_combined': 'data/val_combined.csv',
        'test_combined': 'data/test_combined.csv',
        'train_pb': 'data/train_pb.csv',
        'val_pb': 'data/val_pb.csv',
        'test_pb': 'data/test_pb.csv',
        'train_mb': 'data/train_mb.csv',
        'val_mb': 'data/val_mb.csv',
        'test_mb': 'data/test_mb.csv'
    }

    # Load the datasets
    datasets = load_and_preprocess_data(file_paths)

    # Initialize ensemble models
    rnn_ensemble = RNNEnsemble()
    deep_learning_ensemble = DeepLearningEnsemble()
    reservoir_ensemble = ReservoirEnsemble()
    meta_learner = MetaLearner()

    # Define datasets to process
    dataset_types = ['combined', 'pb', 'mb']

    for dataset_type in dataset_types:
        train_data = datasets.get(f'train_{dataset_type}')
        val_data = datasets.get(f'val_{dataset_type}')
        test_data = datasets.get(f'test_{dataset_type}')

        if train_data is None or val_data is None or test_data is None:
            logging.error(f"Missing data for {dataset_type} dataset. Skipping...")
            continue

        # Train and save RNN ensemble
        logging.info(f"Training RNN ensemble for {dataset_type} dataset...")
        rnn_ensemble.train_ensemble(train_data, val_data)
        rnn_predictions = rnn_ensemble.evaluate_ensemble(test_data)
        logging.info(f"RNN Predictions: {rnn_predictions}")
        rnn_ensemble.save_predictions(rnn_predictions, dataset_type)

        # Train and save Deep Learning ensemble
        logging.info(f"Training Deep Learning ensemble for {dataset_type} dataset...")
        deep_learning_ensemble.train_ensemble(train_data, val_data, dataset_type)
        dl_predictions = deep_learning_ensemble.evaluate_ensemble(test_data)
        logging.info(f"Deep Learning Predictions: {dl_predictions}")
        deep_learning_ensemble.save_predictions(dl_predictions, dataset_type)

        # Train and save Reservoir ensemble
        logging.info(f"Training Reservoir ensemble for {dataset_type} dataset...")
        reservoir_ensemble.train_ensemble(train_data, val_data)
        res_predictions = reservoir_ensemble.evaluate_ensemble(test_data)
        logging.info(f"Reservoir Predictions: {res_predictions}")
        reservoir_ensemble.save_predictions(res_predictions, dataset_type)

        # Aggregate predictions from different models for Meta Learner
        rnn_df = pd.DataFrame(rnn_predictions, columns=['num1_rnn', 'num2_rnn', 'num3_rnn', 'num4_rnn', 'num5_rnn', 'numA_rnn'])
        dl_df = pd.DataFrame(dl_predictions, columns=['num1_dl', 'num2_dl', 'num3_dl', 'num4_dl', 'num5_dl', 'numA_dl'])
        res_df = pd.DataFrame(res_predictions, columns=['num1_res', 'num2_res', 'num3_res', 'num4_res', 'num5_res', 'numA_res'])

        # Log the shapes of individual prediction DataFrames
        logging.info(f"RNN predictions shape: {rnn_df.shape}")
        logging.info(f"Deep Learning predictions shape: {dl_df.shape}")
        logging.info(f"Reservoir predictions shape: {res_df.shape}")

        # Ensure all prediction DataFrames have the same number of rows
        min_length = min(len(rnn_df), len(dl_df), len(res_df))
        if min_length == 0:
            logging.error(f"Aggregated predictions have zero length for {dataset_type}. Skipping...")
            continue
        
        rnn_df = rnn_df.iloc[:min_length]
        dl_df = dl_df.iloc[:min_length]
        res_df = res_df.iloc[:min_length]

        aggregated_predictions = pd.concat([rnn_df, dl_df, res_df], axis=1)
        
        logging.info(f"Aggregated predictions shape: {aggregated_predictions.shape}")

        logging.info(f"Training Meta Learner for {dataset_type} dataset...")
        model = meta_learner.train_meta_learner(aggregated_predictions.values, train_data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].values[:min_length], dataset_type)
        meta_predictions = model.predict(aggregated_predictions.values)
        meta_learner.save_predictions(meta_predictions, 'meta_learner', dataset_type)

        logging.info(f"Completed processing for {dataset_type} dataset.")

    logging.info("Stacked learning process completed.")

if __name__ == "__main__":
    main()
