# meta_learner.py

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MetaLearner:
    def __init__(self):
        self.models = ['lstm', 'gru', 'esn', 'lsm', 'pinn', 'dbn']  # Updated model names

    def load_latest_predictions(self, model_name, dataset_type):
        predictions_dir = 'data/predictions'
        files = [f for f in os.listdir(predictions_dir) if f.startswith(model_name) and dataset_type in f]
        if not files:
            logging.error(f"No prediction files found for {model_name} with type {dataset_type}.")
            return None
        latest_file = max(files, key=lambda x: datetime.strptime(x.split('_')[-1].split('.')[0], '%Y-%m-%d'))
        path = os.path.join(predictions_dir, latest_file)
        logging.info(f"Loading latest predictions from {path}")
        return pd.read_csv(path)

    def prepare_meta_learner_data(self, dataset_type, y_test):
        logging.info(f"Loading predictions for {dataset_type} dataset...")
        predictions = []
        for model in self.models:
            preds = self.load_latest_predictions(model, dataset_type)
            if preds is not None:
                predictions.append(preds)
        
        if not predictions:
            logging.error(f"Missing predictions for {dataset_type} dataset.")
            return None

        combined_predictions = pd.concat(predictions, axis=1)
        X = combined_predictions.iloc[:len(y_test)]

        if X.shape[0] != y_test.shape[0]:
            logging.error(f"Data cardinality mismatch for {dataset_type}: X size {X.shape[0]}, y size {y_test.shape[0]}")
            return None
        
        return X.values

    def create_meta_learner_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(6)  # Assuming 6 outputs corresponding to the 6 numbers to predict
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_meta_learner(self, X_train, y_train, dataset_type):
        input_shape = X_train.shape[1:]
        model = self.create_meta_learner_model(input_shape)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        self.save_model(model, f'stacked/models/meta_learner_{dataset_type}.keras')
        return model

    def save_model(self, model, file_path):
        model.save(file_path)
        logging.info(f"Model saved to {file_path}")

    def save_predictions(self, predictions, model_name, dataset_type):
        today = datetime.today().strftime('%Y-%m-%d')
        predictions_dir = 'data/predictions'
        os.makedirs(predictions_dir, exist_ok=True)
        
        predictions_df = pd.DataFrame(predictions, columns=['num1', 'num2', 'num3', 'num4', 'num5', 'numA'])
        predictions_file = os.path.join(predictions_dir, f'{model_name}_{dataset_type}_predictions_{today}.csv')
        predictions_df.to_csv(predictions_file, index=False)
        logging.info(f"Predictions saved to {predictions_file}")

    def validate_predictions(self, predictions_df):
        invalid_rows = predictions_df[
            (predictions_df[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].sum(axis=1) != predictions_df['numSum']) |
            (predictions_df[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].sum(axis=1) != predictions_df['totalSum'])
        ]
        if not invalid_rows.empty:
            logging.warning(f"Invalid predictions found:\n{invalid_rows}")
        return predictions_df[~predictions_df.index.isin(invalid_rows.index)]

def main():
    meta_learner = MetaLearner()

    logging.info("Preparing data for the meta-learner...")

    for dataset_type in ['combined', 'pb', 'mb']:
        test_file = f'data/test_{dataset_type}.csv'
        if not os.path.exists(test_file):
            logging.error(f"Test file for {dataset_type} dataset does not exist.")
            continue
        
        test_data = pd.read_csv(test_file)
        y_test = test_data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].values

        X = meta_learner.prepare_meta_learner_data(dataset_type, y_test)
        
        if X is None:
            logging.error(f"Skipping {dataset_type} dataset due to missing predictions.")
            continue

        logging.info(f"Training meta-learner for {dataset_type} dataset...")
        model = meta_learner.train_meta_learner(X, y_test, dataset_type)

        logging.info(f"Making predictions for {dataset_type} dataset...")
        predictions = model.predict(X)
        meta_learner.save_predictions(predictions, 'meta_learner', dataset_type)

    logging.info("Meta-learner ensemble process completed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
