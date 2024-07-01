import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeepLearningModel:
    def __init__(self, model_type, input_shape, output_shape):
        self.model_type = model_type
        self.model = self.create_model(input_shape, output_shape)

    def create_model(self, input_shape, output_shape):
        if self.model_type == 'pinn':
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(output_shape)  # Output layer matches the number of target columns
            ])
            model.compile(optimizer='adam', loss=self.pinn_loss)
        elif self.model_type == 'dbn':
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(output_shape)  # Output layer matches the number of target columns
            ])
            model.compile(optimizer='adam', loss='mse')
        else:
            raise ValueError("Invalid model type. Choose either 'pinn' or 'dbn'.")
        return model

    @staticmethod
    def pinn_loss(y_true, y_pred):
        """Custom loss function for the Physics-Informed Neural Network (PINN)."""
        y_true = tf.cast(y_true, tf.float32)  # Ensure y_true is float32
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return mse_loss

    def train(self, features, targets, epochs=50, batch_size=32):
        targets = targets.astype('float32')  # Ensure targets are float32
        self.model.fit(features, targets, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, features):
        return self.model.predict(features)

    def save(self, path):
        self.model.save(path)
        logging.info(f"Model saved to {path}")

class DeepLearningEnsemble:
    def __init__(self):
        self.models = {}

    def prepare_data(self, data):
        features = data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']].copy()
        targets = data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].copy()
        
        features = np.expand_dims(features.values, axis=1)
        targets = targets.values
        
        return features, targets

    def train_ensemble(self, train_data, val_data, dataset_type):
        X_train, y_train = self.prepare_data(train_data)
        X_val, y_val = self.prepare_data(val_data)
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_shape = y_train.shape[1]

        logging.info(f"Training PINN model for {dataset_type} dataset...")
        pinn_model = DeepLearningModel('pinn', input_shape, output_shape)
        pinn_model.train(X_train, y_train, epochs=50, batch_size=32)
        self.models['pinn'] = pinn_model

        logging.info(f"Training DBN model for {dataset_type} dataset...")
        dbn_model = DeepLearningModel('dbn', input_shape, output_shape)
        dbn_model.train(X_train, y_train, epochs=50, batch_size=32)
        self.models['dbn'] = dbn_model

        self.save_models('stacked/models', dataset_type)

    def evaluate_ensemble(self, test_data):
        X_test, _ = self.prepare_data(test_data)
        predictions = {name: model.predict(X_test) for name, model in self.models.items()}
        return predictions

    def save_models(self, directory, dataset_type):
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            model.save(os.path.join(directory, f'{name}_model_{dataset_type}.h5'))

    def validate_predictions(self, predictions_df):
        if 'numSum' not in predictions_df.columns or 'totalSum' not in predictions_df.columns:
            logging.error("Required columns 'numSum' or 'totalSum' are missing in the predictions dataframe.")
            return predictions_df

        predictions_df['predicted_numSum'] = predictions_df[['num1', 'num2', 'num3', 'num4', 'num5']].sum(axis=1)
        predictions_df['predicted_totalSum'] = predictions_df['predicted_numSum'] + predictions_df['numA']

        invalid_rows = predictions_df[
            (predictions_df['predicted_numSum'] != predictions_df['numSum']) |
            (predictions_df['predicted_totalSum'] != predictions_df['totalSum'])
        ]
        if not invalid_rows.empty:
            logging.warning(f"Invalid predictions found:\n{invalid_rows}")
        return predictions_df[~predictions_df.index.isin(invalid_rows.index)]

    def save_predictions(self, predictions, dataset_type):
        today = datetime.today().strftime('%Y-%m-%d')
        predictions_dir = 'data/predictions'
        os.makedirs(predictions_dir, exist_ok=True)
        
        for model_name, preds in predictions.items():
            preds = np.squeeze(preds)  # Remove single-dimensional entries from the shape
            predictions_df = pd.DataFrame(preds, columns=['num1', 'num2', 'num3', 'num4', 'num5', 'numA'])
            
            predictions_df['numSum'] = predictions_df[['num1', 'num2', 'num3', 'num4', 'num5']].sum(axis=1)
            predictions_df['totalSum'] = predictions_df['numSum'] + predictions_df['numA']
            
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

    ensemble = DeepLearningEnsemble()

    logging.info("Training Deep Learning ensemble for combined dataset...")
    ensemble.train_ensemble(train_combined, val_combined, 'combined')
    predictions_combined = ensemble.evaluate_ensemble(test_combined)
    ensemble.save_predictions(predictions_combined, 'combined')

    logging.info("Training Deep Learning ensemble for PB dataset...")
    ensemble.train_ensemble(train_pb, val_pb, 'pb')
    predictions_pb = ensemble.evaluate_ensemble(test_pb)
    ensemble.save_predictions(predictions_pb, 'pb')

    logging.info("Training Deep Learning ensemble for MB dataset...")
    ensemble.train_ensemble(train_mb, val_mb, 'mb')
    predictions_mb = ensemble.evaluate_ensemble(test_mb)
    ensemble.save_predictions(predictions_mb, 'mb')

    logging.info("Loading latest predictions for combined dataset:")
    latest_combined = ensemble.load_latest_predictions('pinn', 'combined')
    print(latest_combined)

    logging.info("Loading latest predictions for PB dataset:")
    latest_pb = ensemble.load_latest_predictions('dbn', 'pb')
    print(latest_pb)

    logging.info("Loading latest predictions for MB dataset:")
    latest_mb = ensemble.load_latest_predictions('pinn', 'mb')
    print(latest_mb)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()