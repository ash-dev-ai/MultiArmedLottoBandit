# stacked_lstm.py

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sqlite3
import pandas as pd
import numpy as np
import json
from features import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

def load_lottery_data(db_file_path='./data/lottery_data.db', table_name='lottery_data'):
    """
    Load lottery data from the database.
    
    Args:
    - db_file_path: Path to the database file
    - table_name: Name of the table containing lottery data
    
    Returns:
    - lottery_data: DataFrame containing lottery data
    """
    try:
        conn = sqlite3.connect(db_file_path)
        lottery_data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return lottery_data
    except Exception as e:
        logging.error(f"Error occurred while loading lottery data: {e}")

def load_combinations_data(db_file_path='./data/all_combinations.db', table_name='all_combinations'):
    """
    Load combinations data from the database.
    
    Args:
    - db_file_path: Path to the database file
    - table_name: Name of the table containing combinations data
    
    Returns:
    - combinations_data: DataFrame containing combinations data
    """
    try:
        conn = sqlite3.connect(db_file_path)
        combinations_data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return combinations_data
    except Exception as e:
        logging.error(f"Error occurred while loading combinations data: {e}")

def data_cleaning(lottery_data, combinations_data):
    """
    Perform data cleaning steps including handling missing values, outliers, and normalization.
    
    Args:
    - lottery_data: DataFrame containing historical lottery data
    - combinations_data: DataFrame containing combinations data
    
    Returns:
    - normalized_lottery_data: Cleaned and normalized lottery data
    - normalized_combinations_data: Cleaned and normalized combinations data
    """
    try:
        # Handling missing values (if necessary)
        # lottery_data.fillna(method='ffill', inplace=True)
        
        # Define a function to remove outliers using IQR method
        def remove_outliers(df):
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

        # Remove outliers from lottery_data
        lottery_data_cleaned = remove_outliers(lottery_data)

        # Normalize data using Min-Max Scaling
        scaler = MinMaxScaler()

        # Normalize the data
        normalized_lottery_data = scaler.fit_transform(lottery_data_cleaned)
        normalized_combinations_data = scaler.fit_transform(combinations_data)

        return normalized_lottery_data, normalized_combinations_data
    except Exception as e:
        logging.error(f"Error occurred during data cleaning: {e}")

def build_lstm_model(input_shape, output_shape, dropout_rate=0.2, units=64):
    """
    Build a stacked LSTM model.
    """
    try:
        model = Sequential([
            LSTM(units=units, input_shape=input_shape, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units=units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units=units),
            Dropout(dropout_rate),
            Dense(units=output_shape, activation='softmax')
        ])
        return model
    except Exception as e:
        logging.error(f"Error occurred while building LSTM model: {e}")

def train_model(X_train, y_train, X_val, y_val, input_shape, output_shape, epochs, batch_size):
    """
    Train the LSTM model on the training data.
    """
    try:
        # Build the LSTM model
        model = build_lstm_model(input_shape, output_shape)
        model.compile(optimizer=Adam(),
                      loss=CategoricalCrossentropy(),
                      metrics=[Accuracy()])

        # Define early stopping callback
        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

        # Define model checkpoint callback to save the best model during training
        checkpoint_path = './model/stacked_lstm/best_model.h5'
        model_checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                           monitor='val_loss',
                                           save_best_only=True,
                                           verbose=1)

        # Train the model
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping, model_checkpoint])

        return model, history
    except Exception as e:
        logging.error(f"Error occurred during model training: {e}")
        return None, None

def continue_training_with_combinations(model, combinations_data, sequence_length, epochs, batch_size):
    """
    Continue training the LSTM model with combinations data.
    """
    try:
        # Prepare sequences and one-hot encoding for combinations data
        X_combinations, y_combinations = prepare_sequences_and_one_hot_encoding(combinations_data, sequence_length)

        # Continue training the model with combinations data
        model.fit(X_combinations, y_combinations, epochs=epochs, batch_size=batch_size)

        return model
    except Exception as e:
        logging.error(f"Error occurred during continuation of training with combinations data: {e}")
        return None

def save_model(model, file_path):
    """
    Save the trained model to a file.
    """
    try:
        model.save(file_path)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Error occurred while saving the model: {e}")

def main():
    try:
        # Configure logging
        logging.basicConfig(filename='stacked_lstm.log', level=logging.INFO)

        # Load data from databases and files
        lottery_data = load_lottery_data()
        combinations_data = load_combinations_data()

        # Perform data preprocessing
        normalized_lottery_data, normalized_combinations_data = data_cleaning(lottery_data, combinations_data)
        
        # Extract features (assuming features.py contains the necessary feature extraction functions)
        lottery_data = extract_date_features(lottery_data)
        lottery_data = extract_historical_patterns(lottery_data)
        lottery_data = extract_statistical_measures(lottery_data)
        number_relationships_features = extract_number_relationships(lottery_data)
        time_series_features = extract_time_series_features(lottery_data)
        lottery_data = extract_combination_characteristics(lottery_data)

        # Prepare sequences and perform one-hot encoding for lottery data
        sequence_length = 10  # Adjust the sequence length as needed
        X, y = prepare_sequences_and_one_hot_encoding(lottery_data, sequence_length)

        # Split the lottery data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=None)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=None)

        # Define input and output shapes
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_shape = y_train.shape[1]
        
        # Train the model on lottery data
        model, history = train_model(X_train, y_train, X_val, y_val, input_shape, output_shape, epochs=100, batch_size=32)

        # Continue training the model with combinations data
        model = continue_training_with_combinations(model, combinations_data, sequence_length, epochs=100, batch_size=32)

        # Save the final model
        final_model_path = './model/stacked_lstm/final_model.h5'
        save_model(model, final_model_path)
    except Exception as e:
        logging.error(f"Error occurred during main execution: {e}")

if __name__ == "__main__":
    main()
