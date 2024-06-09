# deep_learning_ensemble.py

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def pinn_loss(y_true, y_pred):
    """Custom loss function for the Physics-Informed Neural Network (PINN)."""
    y_true = tf.cast(y_true, tf.float32)  # Ensure y_true is float32
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse_loss

def create_pinn_model(input_shape, output_shape):
    """Create and compile the Physics-Informed Neural Network (PINN) model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_shape)  # Output layer matches the number of target columns
    ])
    model.compile(optimizer='adam', loss=pinn_loss)
    return model

def create_dbn_model(input_shape, output_shape):
    """Create and compile the Deep Belief Network (DBN) model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_shape)  # Output layer matches the number of target columns
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_predict_pinn(features, targets, dataset_type):
    """Train the PINN model and make predictions."""
    input_shape = features.shape[1:]
    output_shape = targets.shape[1]  # Adjust output shape to match the number of target columns
    model = create_pinn_model(input_shape, output_shape)
    targets = targets.astype('float32')  # Ensure targets are float32
    model.fit(features, targets, epochs=50, batch_size=32, verbose=0)
    predictions = model.predict(features)
    save_model(model, f'stacked/models/pinn_model_{dataset_type}.h5')
    return predictions

def train_and_predict_dbn(features, targets, dataset_type):
    """Train the DBN model and make predictions."""
    input_shape = features.shape[1:]
    output_shape = targets.shape[1]  # Adjust output shape to match the number of target columns
    model = create_dbn_model(input_shape, output_shape)
    targets = targets.astype('float32')  # Ensure targets are float32
    model.fit(features, targets, epochs=50, batch_size=32, verbose=0)
    predictions = model.predict(features)
    save_model(model, f'stacked/models/dbn_model_{dataset_type}.h5')
    return predictions

def save_model(model, file_path):
    """Save the model to the specified file path."""
    model.save(file_path)
    logging.info(f"Model saved to {file_path}")

def save_predictions(predictions, dataset_name, dataset_type):
    """Save the predictions to a CSV file."""
    today = datetime.today().strftime('%Y-%m-%d')
    predictions_dir = 'data/predictions'
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    
    predictions_df = pd.DataFrame(predictions, columns=['num1', 'num2', 'num3', 'num4', 'num5', 'numA'])
    predictions_file = os.path.join(predictions_dir, f'deep_learning_{dataset_type}_predictions_{today}.csv')
    predictions_df.to_csv(predictions_file, index=False)
    logging.info(f"Predictions saved to {predictions_file}")

def main():
    # Load datasets
    train_combined = pd.read_csv('data/train_combined.csv')
    val_combined = pd.read_csv('data/val_combined.csv')
    test_combined = pd.read_csv('data/test_combined.csv')

    train_pb = pd.read_csv('data/train_pb.csv')
    val_pb = pd.read_csv('data/val_pb.csv')
    test_pb = pd.read_csv('data/test_pb.csv')

    train_mb = pd.read_csv('data/train_mb.csv')
    val_mb = pd.read_csv('data/val_mb.csv')
    test_mb = pd.read_csv('data/test_mb.csv')

    # Prepare features and targets for combined dataset
    features_combined = train_combined[['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']].values
    targets_combined = train_combined[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].values

    # Train and predict with PINN for combined dataset
    pinn_predictions_combined = train_and_predict_pinn(features_combined, targets_combined, "combined")
    save_predictions(pinn_predictions_combined, 'combined', 'combined')

    # Train and predict with DBN for combined dataset
    dbn_predictions_combined = train_and_predict_dbn(features_combined, targets_combined, "combined")
    save_predictions(dbn_predictions_combined, 'combined', 'combined')

    # PB Dataset
    features_pb = train_pb[['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']].values
    targets_pb = train_pb[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].values

    pinn_predictions_pb = train_and_predict_pinn(features_pb, targets_pb, "pb")
    save_predictions(pinn_predictions_pb, 'pb', 'pb')

    dbn_predictions_pb = train_and_predict_dbn(features_pb, targets_pb, "pb")
    save_predictions(dbn_predictions_pb, 'pb', 'pb')

    # MB Dataset
    features_mb = train_mb[['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']].values
    targets_mb = train_mb[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].values

    pinn_predictions_mb = train_and_predict_pinn(features_mb, targets_mb, "mb")
    save_predictions(pinn_predictions_mb, 'mb', 'mb')

    dbn_predictions_mb = train_and_predict_dbn(features_mb, targets_mb, "mb")
    save_predictions(dbn_predictions_mb, 'mb', 'mb')

    logging.info("Deep learning models have been trained and predictions saved.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
