import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = models.Sequential([
        layers.LSTM(64, activation='tanh', input_shape=input_shape, return_sequences=True),
        layers.LSTM(64, activation='tanh'),
        layers.Dense(6)  # Output layer for 6 predictions
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to create GRU model
def create_gru_model(input_shape):
    model = models.Sequential([
        layers.GRU(64, activation='tanh', input_shape=input_shape, return_sequences=True),
        layers.GRU(64, activation='tanh'),
        layers.Dense(6)  # Output layer for 6 predictions
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Prepare the data
def prepare_rnn_data(data):
    """Prepare features for the RNN models."""
    features = data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']].copy()
    targets = data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].copy()
    
    # Convert to 3D shape (samples, timesteps, features)
    features = np.expand_dims(features.values, axis=1)
    targets = targets.values
    
    return features, targets

def train_rnn_ensemble(train_data, val_data, dataset_type):
    X_train, y_train = prepare_rnn_data(train_data)
    X_val, y_val = prepare_rnn_data(val_data)
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Train LSTM model
    lstm_model = create_lstm_model(input_shape)
    lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0)
    
    # Train GRU model
    gru_model = create_gru_model(input_shape)
    gru_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0)

    models = {
        'lstm': lstm_model,
        'gru': gru_model
    }

    save_models(models, f'stacked/models')

    return models

def evaluate_rnn_ensemble(models, test_data):
    X_test, _ = prepare_rnn_data(test_data)
    predictions = {name: model.predict(X_test) for name, model in models.items()}
    return predictions

def save_models(models, directory):
    os.makedirs(directory, exist_ok=True)
    for name, model in models.items():
        model.save(os.path.join(directory, f'{name}_model.h5'))

def save_predictions(predictions, dataset_name, dataset_type):
    today = datetime.today().strftime('%Y-%m-%d')
    predictions_dir = 'data/predictions'
    os.makedirs(predictions_dir, exist_ok=True)
    
    for model_name, preds in predictions.items():
        predictions_df = pd.DataFrame(preds, columns=['num1', 'num2', 'num3', 'num4', 'num5', 'numA'])
        predictions_file = os.path.join(predictions_dir, f'{model_name}_{dataset_type}_predictions_{today}.csv')
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

    # Train and evaluate RNN ensemble for each dataset type
    logging.info("Training RNN ensemble for combined dataset...")
    models_combined = train_rnn_ensemble(train_combined, val_combined, 'combined')
    predictions_combined = evaluate_rnn_ensemble(models_combined, test_combined)
    save_predictions(predictions_combined, 'combined', 'combined')

    logging.info("Training RNN ensemble for PB dataset...")
    models_pb = train_rnn_ensemble(train_pb, val_pb, 'pb')
    predictions_pb = evaluate_rnn_ensemble(models_pb, test_pb)
    save_predictions(predictions_pb, 'pb', 'pb')

    logging.info("Training RNN ensemble for MB dataset...")
    models_mb = train_rnn_ensemble(train_mb, val_mb, 'mb')
    predictions_mb = evaluate_rnn_ensemble(models_mb, test_mb)
    save_predictions(predictions_mb, 'mb', 'mb')

    logging.info("Predictions for combined dataset:")
    print(predictions_combined)

    logging.info("Predictions for PB dataset:")
    print(predictions_pb)

    logging.info("Predictions for MB dataset:")
    print(predictions_mb)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
