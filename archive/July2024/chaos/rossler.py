# rossler.py
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime

class RosslerAttractor:
    def __init__(self, a=0.2, b=0.2, c=5.7, dt=0.01, steps=10000):
        self.a = a
        self.b = b
        self.c = c
        self.dt = dt
        self.steps = steps

    def generate_chaos_data(self):
        x, y, z = 0, 1, 1.05
        data = []

        def rossler_dynamics(x, y, z, a, b, c):
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)
            return dx, dy, dz

        for _ in range(self.steps):
            dx, dy, dz = rossler_dynamics(x, y, z, self.a, self.b, self.c)
            x += dx * self.dt
            y += dy * self.dt
            z += dz * self.dt
            data.append((x, y, z))

        return np.array(data)

    def transform_with_chaos(self, data):
        num_rows = len(data)
        chaos_data = self.generate_chaos_data()
        # Ensure the chaos data has the same number of rows as the original data
        chaos_data = chaos_data[:num_rows]
        chaos_df = pd.DataFrame(chaos_data, columns=['rossler_x', 'rossler_y', 'rossler_z'])
        transformed_data = pd.concat([data.reset_index(drop=True), chaos_df], axis=1)
        if transformed_data.isna().sum().sum() > 0:
            logging.warning(f'Transformed data contains NaNs: \n{transformed_data.isna().sum()}')
        return transformed_data

class ModelTrainer:
    def __init__(self, target_columns):
        self.target_columns = target_columns
        self.models = {}

    def train_model(self, train_data, val_data):
        for target_column in self.target_columns:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            X_train = train_data.drop(columns=self.target_columns + ['date'])
            y_train = train_data[target_column]
            X_val = val_data.drop(columns=self.target_columns + ['date'])
            y_val = val_data[target_column]

            if X_train.isna().sum().sum() > 0:
                logging.warning(f'X_train contains NaNs: \n{X_train.isna().sum()}')
            if X_val.isna().sum().sum() > 0:
                logging.warning(f'X_val contains NaNs: \n{X_val.isna().sum()}')

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            logging.info(f'Validation MSE for {target_column}: {mse}')
            self.models[target_column] = model

    def evaluate_model(self, test_data):
        predictions = {}
        for target_column in self.target_columns:
            model = self.models[target_column]
            X_test = test_data.drop(columns=self.target_columns + ['date'])
            y_test = test_data[target_column]
            if X_test.isna().sum().sum() > 0:
                logging.warning(f'X_test contains NaNs: \n{X_test.isna().sum()}')

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            logging.info(f'Test MSE for {target_column}: {mse}')
            predictions[target_column] = y_pred
        return predictions

class PredictionSaver:
    @staticmethod
    def save_predictions(predictions, model_name):
        date_str = datetime.now().strftime("%Y-%m-%d")
        for dataset, dataset_name in zip(predictions, ["combined", "pb", "mb"]):
            filepath = f"data/predictions/{model_name}_predictions_{dataset_name}_{date_str}.csv"
            logging.info(f"Saving {dataset_name} predictions to {filepath}")
            dataset.to_csv(filepath, index=False)

def run_rossler(return_predictions=False):
    logging.info("Loading datasets...")
    train_combined = pd.read_csv('data/train_combined.csv')
    val_combined = pd.read_csv('data/val_combined.csv')
    test_combined = pd.read_csv('data/test_combined.csv')
    
    train_pb = pd.read_csv('data/train_pb.csv')
    val_pb = pd.read_csv('data/val_pb.csv')
    test_pb = pd.read_csv('data/test_pb.csv')
    
    train_mb = pd.read_csv('data/train_mb.csv')
    val_mb = pd.read_csv('data/val_mb.csv')
    test_mb = pd.read_csv('data/test_mb.csv')
    
    # Use specified columns
    columns_to_use = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']
    train_combined = train_combined[columns_to_use]
    val_combined = val_combined[columns_to_use]
    test_combined = test_combined[columns_to_use]
    
    train_pb = train_pb[columns_to_use]
    val_pb = val_pb[columns_to_use]
    test_pb = test_pb[columns_to_use]
    
    train_mb = train_mb[columns_to_use]
    val_mb = val_mb[columns_to_use]
    test_mb = test_mb[columns_to_use]
    
    logging.info("Transforming datasets with Rössler Attractor...")
    rossler = RosslerAttractor(steps=len(train_combined))
    train_combined = rossler.transform_with_chaos(train_combined)
    val_combined = rossler.transform_with_chaos(val_combined)
    test_combined = rossler.transform_with_chaos(test_combined)
    
    rossler = RosslerAttractor(steps=len(train_pb))
    train_pb = rossler.transform_with_chaos(train_pb)
    val_pb = rossler.transform_with_chaos(val_pb)
    test_pb = rossler.transform_with_chaos(test_pb)
    
    rossler = RosslerAttractor(steps=len(train_mb))
    train_mb = rossler.transform_with_chaos(train_mb)
    val_mb = rossler.transform_with_chaos(val_mb)
    test_mb = rossler.transform_with_chaos(test_mb)
    
    # Define target columns
    target_columns = ['numSum', 'totalSum']
    
    # Train and evaluate the model
    logging.info(f"Training models with combined dataset for {target_columns}...")
    trainer_combined = ModelTrainer(target_columns)
    trainer_combined.train_model(train_combined, val_combined)
    logging.info(f"Evaluating models with combined test dataset for {target_columns}...")
    predictions_combined = trainer_combined.evaluate_model(test_combined)
    
    logging.info(f"Training models with PB dataset for {target_columns}...")
    trainer_pb = ModelTrainer(target_columns)
    trainer_pb.train_model(train_pb, val_pb)
    logging.info(f"Evaluating models with PB test dataset for {target_columns}...")
    predictions_pb = trainer_pb.evaluate_model(test_pb)
    
    logging.info(f"Training models with MB dataset for {target_columns}...")
    trainer_mb = ModelTrainer(target_columns)
    trainer_mb.train_model(train_mb, val_mb)
    logging.info(f"Evaluating models with MB test dataset for {target_columns}...")
    predictions_mb = trainer_mb.evaluate_model(test_mb)
    
    # Save predictions
    PredictionSaver.save_predictions([pd.DataFrame(predictions_combined), pd.DataFrame(predictions_pb), pd.DataFrame(predictions_mb)], "rossler")
    
    if return_predictions:
        return predictions_combined, predictions_pb, predictions_mb

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_rossler()
