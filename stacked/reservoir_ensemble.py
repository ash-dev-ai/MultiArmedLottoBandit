# reservoir_ensemble.py

import numpy as np
import pandas as pd
from datetime import datetime
import os
import logging
import pickle
from sklearn.linear_model import Ridge
from stacked.libs.pyESN.pyESN import ESN
from reservoirpy.nodes import Reservoir, Ridge as LSMRidge  # Liquid State Machine implementation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ReservoirModel:
    def __init__(self, model_type, n_inputs=None, n_outputs=None, reservoir_size=500):
        self.model_type = model_type
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.reservoir_size = reservoir_size
        self.model = self.create_model()

    def create_model(self):
        if self.model_type == 'esn':
            return ESN(
                n_inputs=self.n_inputs,
                n_outputs=self.n_outputs,
                n_reservoir=self.reservoir_size,
                sparsity=0.2,
                random_state=42,
                spectral_radius=0.95
            )
        elif self.model_type == 'lsm':
            reservoir = Reservoir(units=self.reservoir_size)
            reservoir.spectral_radius = 0.95
            return reservoir >> LSMRidge()
        else:
            raise ValueError("Invalid model type. Choose either 'esn' or 'lsm'.")

    def train(self, X_train, y_train):
        if self.model_type == 'esn':
            self.model.fit(X_train, y_train)
        elif self.model_type == 'lsm':
            self.model = self.model.fit(X_train, y_train)

    def predict(self, X):
        if self.model_type == 'esn':
            return self.model.predict(X)
        elif self.model_type == 'lsm':
            return self.model.run(X)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        if self.model_type == 'esn':
            np.save(os.path.join(path, 'W_in.npy'), self.model.W_in)
            np.save(os.path.join(path, 'W.npy'), self.model.W)
            np.save(os.path.join(path, 'W_out.npy'), self.model.W_out)
        elif self.model_type == 'lsm':
            with open(os.path.join(path, 'model.pkl'), 'wb') as f:
                pickle.dump(self.model, f)

    @staticmethod
    def load(path, model_type, n_inputs=None, n_outputs=None, reservoir_size=500):
        if model_type == 'esn':
            W_in = np.load(os.path.join(path, 'W_in.npy'))
            W = np.load(os.path.join(path, 'W.npy'))
            W_out = np.load(os.path.join(path, 'W_out.npy'))
            model = ESN(
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                n_reservoir=reservoir_size,
                sparsity=0.2,
                random_state=42,
                spectral_radius=0.95
            )
            model.W_in = W_in
            model.W = W
            model.W_out = W_out
        elif model_type == 'lsm':
            with open(os.path.join(path, 'model.pkl'), 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError("Invalid model type. Choose either 'esn' or 'lsm'.")
        return ReservoirModel(model_type, n_inputs, n_outputs, reservoir_size), model

class ReservoirEnsemble:
    def __init__(self):
        self.models = {}

    def prepare_data(self, data):
        X = data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']].copy()
        y = data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum']].copy()
        X = (X - X.mean()) / X.std()
        return X.values, y.values

    def train_ensemble(self, train_data, val_data):
        X_train, y_train = self.prepare_data(train_data)
        X_val, y_val = self.prepare_data(val_data)

        esn_model = ReservoirModel('esn', n_inputs=X_train.shape[1], n_outputs=y_train.shape[1])
        esn_model.train(X_train, y_train)
        self.models['esn'] = esn_model

        lsm_model = ReservoirModel('lsm', n_inputs=X_train.shape[1], n_outputs=y_train.shape[1])
        lsm_model.train(X_train, y_train)
        self.models['lsm'] = lsm_model

        self.save_models('stacked/models')

    def evaluate_ensemble(self, test_data):
        X_test, _ = self.prepare_data(test_data)
        predictions = {name: model.predict(X_test) for name, model in self.models.items()}
        return predictions

    def save_models(self, directory):
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            model.save(os.path.join(directory, name))

    def validate_predictions(self, predictions_df):
        invalid_rows = predictions_df[
            (predictions_df[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].sum(axis=1) != predictions_df['numSum']) |
            (predictions_df[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].sum(axis=1) != predictions_df['totalSum'])
        ]
        if not invalid_rows.empty:
            logging.warning(f"Invalid predictions found:\n{invalid_rows}")
        return predictions_df[~predictions_df.index.isin(invalid_rows.index)]

    def save_predictions(self, predictions, dataset_type):
        today = datetime.today().strftime('%Y-%m-%d')
        predictions_dir = 'data/predictions'
        os.makedirs(predictions_dir, exist_ok=True)
        
        for model_name, preds in predictions.items():
            predictions_df = pd.DataFrame(preds, columns=['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum'])
            predictions_df = self.validate_predictions(predictions_df)
            predictions_file = os.path.join(predictions_dir, f'{model_name}_{dataset_type}_predictions_{today}.csv')
            predictions_df.to_csv(predictions_file, index=False)
            logging.info(f"Predictions saved to {predictions_file}")

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

def main():
    train_combined = pd.read_csv('data/train_combined.csv')
    val_combined = pd.read_csv('data/val_combined.csv')
    test_combined = pd.read_csv('data/test_combined.csv')

    train_pb = pd.read_csv('data/train_pb.csv')
    val_pb = pd.read_csv('data/val_pb.csv')
    test_pb = pd.read_csv('data/test_pb.csv')

    train_mb = pd.read_csv('data/train_mb.csv')
    val_mb = pd.read_csv('data/val_mb.csv')
    test_mb = pd.read_csv('data/test_mb.csv')

    ensemble = ReservoirEnsemble()

    logging.info("Training Reservoir ensemble for combined dataset...")
    ensemble.train_ensemble(train_combined, val_combined)
    predictions_combined = ensemble.evaluate_ensemble(test_combined)
    ensemble.save_predictions(predictions_combined, 'combined')

    logging.info("Training Reservoir ensemble for PB dataset...")
    ensemble.train_ensemble(train_pb, val_pb)
    predictions_pb = ensemble.evaluate_ensemble(test_pb)
    ensemble.save_predictions(predictions_pb, 'pb')

    logging.info("Training Reservoir ensemble for MB dataset...")
    ensemble.train_ensemble(train_mb, val_mb)
    predictions_mb = ensemble.evaluate_ensemble(test_mb)
    ensemble.save_predictions(predictions_mb, 'mb')

    logging.info("Loading latest predictions for combined dataset:")
    latest_combined = ensemble.load_latest_predictions('esn', 'combined')
    print(latest_combined)

    logging.info("Loading latest predictions for PB dataset:")
    latest_pb = ensemble.load_latest_predictions('lsm', 'pb')
    print(latest_pb)

    logging.info("Loading latest predictions for MB dataset:")
    latest_mb = ensemble.load_latest_predictions('esn', 'mb')
    print(latest_mb)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main
()
