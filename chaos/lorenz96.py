# lorenz96.py
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime

def lorenz96(F=8, dt=0.01, steps=10000, n=36):
    x = F * np.ones(n)
    x[19] += 0.01  # Add small perturbation
    data = []

    def lorenz96_dynamics(x, F):
        d = np.zeros(n)
        for i in range(n):
            d[i] = (x[(i + 1) % n] - x[i - 2]) * x[i - 1] - x[i] + F
        return d

    for _ in range(steps):
        dx = lorenz96_dynamics(x, F)
        x += dx * dt
        data.append(x.copy())

    return np.array(data)

def transform_with_lorenz96(data):
    chaos_data = lorenz96(steps=len(data))
    chaos_df = pd.DataFrame(chaos_data, columns=[f'lorenz96_{i}' for i in range(chaos_data.shape[1])])
    return pd.concat([data.reset_index(drop=True), chaos_df], axis=1)

def train_model(train_data, val_data, target_columns):
    models = {}
    for target_column in target_columns:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X_train = train_data.drop(columns=target_columns + ['draw_date'])
        y_train = train_data[target_column]
        X_val = val_data.drop(columns=target_columns + ['draw_date'])
        y_val = val_data[target_column]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        logging.info(f'Validation MSE for {target_column}: {mse}')
        models[target_column] = model
    return models

def evaluate_model(models, test_data, target_columns):
    predictions = {}
    for target_column in target_columns:
        model = models[target_column]
        X_test = test_data.drop(columns=target_columns + ['draw_date'])
        y_test = test_data[target_column]
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f'Test MSE for {target_column}: {mse}')
        predictions[target_column] = y_pred
    return predictions

def save_predictions(predictions, model_name):
    date_str = datetime.now().strftime("%Y-%m-%d")
    for dataset, dataset_name in zip(predictions, ["combined", "pb", "mb"]):
        filepath = f"data/predictions/{model_name}_predictions_{dataset_name}_{date_str}.csv"
        logging.info(f"Saving {dataset_name} predictions to {filepath}")
        dataset.to_csv(filepath, index=False)

def run_lorenz96(return_predictions=False):
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
    columns_to_use = ['draw_date', 'num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day']
    train_combined = train_combined[columns_to_use]
    val_combined = val_combined[columns_to_use]
    test_combined = test_combined[columns_to_use]
    
    train_pb = train_pb[columns_to_use]
    val_pb = val_pb[columns_to_use]
    test_pb = test_pb[columns_to_use]
    
    train_mb = train_mb[columns_to_use]
    val_mb = val_mb[columns_to_use]
    test_mb = test_mb[columns_to_use]
    
    logging.info("Transforming datasets with Lorenz96 Model...")
    train_combined = transform_with_lorenz96(train_combined)
    val_combined = transform_with_lorenz96(val_combined)
    test_combined = transform_with_lorenz96(test_combined)
    
    train_pb = transform_with_lorenz96(train_pb)
    val_pb = transform_with_lorenz96(val_pb)
    test_pb = transform_with_lorenz96(test_pb)
    
    train_mb = transform_with_lorenz96(train_mb)
    val_mb = transform_with_lorenz96(val_mb)
    test_mb = transform_with_lorenz96(test_mb)
    
    # Define target columns
    target_columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA']
    
    # Train and evaluate the model
    logging.info(f"Training models with combined dataset for {target_columns}...")
    models_combined = train_model(train_combined, val_combined, target_columns)
    logging.info(f"Evaluating models with combined test dataset for {target_columns}...")
    predictions_combined = evaluate_model(models_combined, test_combined, target_columns)
    
    logging.info(f"Training models with PB dataset for {target_columns}...")
    models_pb = train_model(train_pb, val_pb, target_columns)
    logging.info(f"Evaluating models with PB test dataset for {target_columns}...")
    predictions_pb = evaluate_model(models_pb, test_pb, target_columns)
    
    logging.info(f"Training models with MB dataset for {target_columns}...")
    models_mb = train_model(train_mb, val_mb, target_columns)
    logging.info(f"Evaluating models with MB test dataset for {target_columns}...")
    predictions_mb = evaluate_model(models_mb, test_mb, target_columns)
    
    # Save predictions
    save_predictions([pd.DataFrame(predictions_combined), pd.DataFrame(predictions_pb), pd.DataFrame(predictions_mb)], "lorenz96")
    
    if return_predictions:
        return predictions_combined, predictions_pb, predictions_mb

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_lorenz96()
