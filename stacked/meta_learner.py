# meta_learner.py
# meta_learner.py

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import os
import logging

def load_predictions(model_name, dataset_type):
    """Load predictions from CSV files."""
    today = datetime.today().strftime('%Y-%m-%d')
    predictions_dir = 'data/predictions'
    file_path = os.path.join(predictions_dir, f'{model_name}_{dataset_type}_predictions_{today}.csv')
    
    if os.path.exists(file_path):
        logging.info(f"Loading predictions from {file_path}")
        return pd.read_csv(file_path)
    else:
        logging.error(f"No prediction files found for {model_name} and dataset type {dataset_type}.")
        return None

def prepare_meta_learner_data(dataset_type, y_test):
    """Prepare data for the meta-learner model."""
    logging.info(f"Loading predictions for {dataset_type} dataset...")
    models = ['lstm', 'gru', 'esn', 'lsm', 'deep_learning']  # Updated model names
    
    predictions = []
    for model in models:
        preds = load_predictions(model, dataset_type)
        if preds is not None:
            predictions.append(preds)
    
    if not predictions:
        logging.error(f"Missing predictions for {dataset_type} dataset.")
        return None

    # Concatenate predictions along columns
    combined_predictions = pd.concat(predictions, axis=1)
    
    # Adjust to match the size of y_test
    X = combined_predictions.iloc[:len(y_test)]

    if X.shape[0] != y_test.shape[0]:
        logging.error(f"Data cardinality mismatch for {dataset_type}: X size {X.shape[0]}, y size {y_test.shape[0]}")
        return None
    
    return X.values

def create_meta_learner_model(input_shape):
    """Create and compile the meta-learner model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(6)  # Assuming 6 outputs corresponding to the 6 numbers to predict
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_meta_learner(X_train, y_train, dataset_type):
    """Train the meta-learner model."""
    input_shape = X_train.shape[1:]
    model = create_meta_learner_model(input_shape)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    save_model(model, f'stacked/models/meta_learner_{dataset_type}.keras')
    return model

def save_model(model, file_path):
    """Save the model to the specified file path."""
    model.save(file_path)
    logging.info(f"Model saved to {file_path}")

def save_predictions(predictions, model_name, dataset_type):
    """Save the predictions to a CSV file."""
    today = datetime.today().strftime('%Y-%m-%d')
    predictions_dir = 'data/predictions'
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    
    predictions_df = pd.DataFrame(predictions, columns=['num1', 'num2', 'num3', 'num4', 'num5', 'numA'])
    predictions_file = os.path.join(predictions_dir, f'{model_name}_{dataset_type}_predictions_{today}.csv')
    predictions_df.to_csv(predictions_file, index=False)
    logging.info(f"Predictions saved to {predictions_file}")

def main():
    logging.info("Preparing data for the meta-learner...")

    for dataset_type in ['combined', 'pb', 'mb']:
        # Load the corresponding test set
        test_file = f'data/test_{dataset_type}.csv'
        if not os.path.exists(test_file):
            logging.error(f"Test file for {dataset_type} dataset does not exist.")
            continue
        
        test_data = pd.read_csv(test_file)
        y_test = test_data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].values

        X = prepare_meta_learner_data(dataset_type, y_test)
        
        if X is None:
            logging.error(f"Skipping {dataset_type} dataset due to missing predictions.")
            continue

        logging.info(f"Training meta-learner for {dataset_type} dataset...")
        model = train_meta_learner(X, y_test, dataset_type)

        logging.info(f"Making predictions for {dataset_type} dataset...")
        predictions = model.predict(X)
        save_predictions(predictions, 'meta_learner', dataset_type)

    logging.info("Meta-learner ensemble process completed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
