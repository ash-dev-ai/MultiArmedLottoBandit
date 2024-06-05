# chua.py
# chua.py
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def chua_circuit(a=15.6, b=28.0, m0=-1.143, m1=-0.714, dt=0.01, steps=10000):
    x, y, z = 0.7, 0, 0
    data = []

    def chua_dynamics(x, y, z, a, b, m0, m1):
        hx = m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))
        dx = a * (y - x - hx)
        dy = x - y + z
        dz = -b * y
        return dx, dy, dz

    for _ in range(steps):
        dx, dy, dz = chua_dynamics(x, y, z, a, b, m0, m1)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        data.append((x, y, z))

    return np.array(data)

def transform_with_chua(data):
    chaos_data = chua_circuit(steps=len(data))
    chaos_df = pd.DataFrame(chaos_data, columns=['chua_x', 'chua_y', 'chua_z'])
    return pd.concat([data.reset_index(drop=True), chaos_df], axis=1)

def train_model(train_data, val_data, target_column):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X_train = train_data.drop(columns=[target_column, 'draw_date'])
    y_train = train_data[target_column]
    X_val = val_data.drop(columns=[target_column, 'draw_date'])
    y_val = val_data[target_column]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    logging.info(f'Validation MSE for {target_column}: {mse}')
    return model

def evaluate_model(model, test_data, target_column):
    X_test = test_data.drop(columns=[target_column, 'draw_date'])
    y_test = test_data[target_column]
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f'Test MSE for {target_column}: {mse}')
    return y_pred

def run_chua(return_predictions=False):
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
    test_mb = val_mb[columns_to_use]
    
    logging.info("Transforming datasets with Chua's Circuit...")
    train_combined = transform_with_chua(train_combined)
    val_combined = transform_with_chua(val_combined)
    test_combined = transform_with_chua(test_combined)
    
    train_pb = transform_with_chua(train_pb)
    val_pb = transform_with_chua(val_pb)
    test_pb = transform_with_chua(test_pb)
    
    train_mb = transform_with_chua(train_mb)
    val_mb = transform_with_chua(val_mb)
    test_mb = transform_with_chua(test_mb)
    
    # Define target columns (example: num1, num2, num3, num4, num5, numA)
    target_columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA']
    
    predictions_combined = {}
    predictions_pb = {}
    predictions_mb = {}
    
    for target_column in target_columns:
        # Train and evaluate the model
        logging.info(f"Training model with combined dataset for {target_column}...")
        model_combined = train_model(train_combined, val_combined, target_column)
        logging.info(f"Evaluating model with combined test dataset for {target_column}...")
        predictions_combined[target_column] = evaluate_model(model_combined, test_combined, target_column)
        
        logging.info(f"Training model with PB dataset for {target_column}...")
        model_pb = train_model(train_pb, val_pb, target_column)
        logging.info(f"Evaluating model with PB test dataset for {target_column}...")
        predictions_pb[target_column] = evaluate_model(model_pb, test_pb, target_column)
        
        logging.info(f"Training model with MB dataset for {target_column}...")
        model_mb = train_model(train_mb, val_mb, target_column)
        logging.info(f"Evaluating model with MB test dataset for {target_column}...")
        predictions_mb[target_column] = evaluate_model(model_mb, test_mb, target_column)
    
    if return_predictions:
        return predictions_combined, predictions_pb, predictions_mb

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_chua()
