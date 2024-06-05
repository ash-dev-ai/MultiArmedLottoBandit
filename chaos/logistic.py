# logistic.py
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def logistic_map(r=3.8, x0=0.5, steps=10000):
    x = x0
    data = []

    for _ in range(steps):
        x = r * x * (1 - x)
        data.append(x)

    return np.array(data)

def transform_with_logistic(data):
    chaos_data = logistic_map(steps=len(data))
    chaos_df = pd.DataFrame(chaos_data, columns=['logistic_x'])
    return pd.concat([data.reset_index(drop=True), chaos_df], axis=1)

def train_model(train_data, val_data, target_column):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_val = val_data.drop(columns=[target_column])
    y_val = val_data[target_column]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    logging.info(f'Validation MSE for {target_column}: {mse}')
    return model

def evaluate_model(model, test_data, target_column):
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f'Test MSE for {target_column}: {mse}')
    return y_pred

def run_logistic(return_predictions=False):
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
    columns_to_use = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day']
    train_combined = train_combined[columns_to_use]
    val_combined = val_combined[columns_to_use]
    test_combined = test_combined[columns_to_use]
    
    train_pb = train_pb[columns_to_use]
    val_pb = val_pb[columns_to_use]
    test_pb = test_pb[columns_to_use]
    
    train_mb = train_mb[columns_to_use]
    val_mb = val_mb[columns_to_use]
    test_mb = test_mb[columns_to_use]
    
    logging.info("Transforming datasets with Logistic Map...")
    train_combined = transform_with_logistic(train_combined)
    val_combined = transform_with_logistic(val_combined)
    test_combined = transform_with_logistic(test_combined)
    
    train_pb = transform_with_logistic(train_pb)
    val_pb = transform_with_logistic(val_pb)
    test_pb = transform_with_logistic(test_pb)
    
    train_mb = transform_with_logistic(train_mb)
    val_mb = transform_with_logistic(val_mb)
    test_mb = transform_with_logistic(test_mb)
    
    # Define target column (example: num3)
    target_column = 'num4'
    
    # Train and evaluate the model
    logging.info(f"Training model with combined dataset for {target_column}...")
    model_combined = train_model(train_combined, val_combined, target_column)
    logging.info(f"Evaluating model with combined test dataset for {target_column}...")
    predictions_combined = evaluate_model(model_combined, test_combined, target_column)
    
    logging.info(f"Training model with PB dataset for {target_column}...")
    model_pb = train_model(train_pb, val_pb, target_column)
    logging.info(f"Evaluating model with PB test dataset for {target_column}...")
    predictions_pb = evaluate_model(model_pb, test_pb, target_column)
    
    logging.info(f"Training model with MB dataset for {target_column}...")
    model_mb = train_model(train_mb, val_mb, target_column)
    logging.info(f"Evaluating model with MB test dataset for {target_column}...")
    predictions_mb = evaluate_model(model_mb, test_mb, target_column)
    
    if return_predictions:
        return predictions_combined, predictions_pb, predictions_mb

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_logistic()

