import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tcn import TCN, tcn_full_summary
from tensorflow.keras.optimizers import Adam
import os
import logging
import pickle
import json

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_latest_data_directory():
    try:
        base_path = 'data/prep'
        latest_date = sorted(os.listdir(base_path), reverse=True)[0]
        latest_time = sorted(os.listdir(f"{base_path}/{latest_date}"), reverse=True)[0]
        return os.path.join(base_path, latest_date, latest_time)
    except Exception as e:
        logging.error(f"Error in get_latest_data_directory: {e}")
        raise

# Add this function to load statistics from the JSON file
def load_statistics(stats_path):
    try:
        with open(stats_path, 'r') as f:
            statistics = json.load(f)
        return statistics
    except Exception as e:
        logging.error(f"Error in load_statistics: {e}")
        raise

def load_latest_dataset():
    try:
        directory_path = get_latest_data_directory()
        dataset_path = os.path.join(directory_path, 'org_df.csv')
        df = pd.read_csv(dataset_path)

        return df
    except Exception as e:
        logging.error(f"Error in load_latest_dataset: {e}")
        raise

def save_model(model, filename):
    try:
        directory_path = get_latest_data_directory()
        # Assuming you're saving a Keras model, adjust the filename to end with '.h5' for clarity
        model_path = os.path.join(directory_path, filename + '.h5')  
        model.save(model_path)  # Saving the Keras model
        logging.info(f"Model saved as {model_path}")
    except Exception as e:
        logging.error(f"Error in save_model: {e}")
        raise


def train_model(df):
    try:
        # Assuming the same feature selection, adjust as necessary
        X = df[['winning_numbers_sum', 'w1', 'w2', 'w3', 'w4', 'w5', 'mega_ball', 'total_sum']].values

        y = df['d'].values

        # Assuming each row is a time step; adjust according to your data structure
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)

        # Define the TCN model
        model = Sequential([
            TCN(input_shape=(X_train.shape[1], X_train.shape[2]), 
                nb_filters=64,
                kernel_size=2,
                dilations=[1, 2, 4],
                padding='causal',
                use_skip_connections=True,
                dropout_rate=0.0,
                return_sequences=False,
                activation='relu',
                kernel_initializer='he_normal',
                use_batch_norm=False,
                use_layer_norm=False,
                use_weight_norm=True),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Train the TCN model
        model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test))

        logging.info("Model training completed.")

        # Note: Model evaluation and saving steps would follow here as needed
        return model
    except Exception as e:
        logging.error(f"Error in train_model: {e}")
        raise

def main():
    try:
        logging.info("Loading datasets...")
        historical_data = load_latest_dataset()

        # Load statistics
        stats_path = f"{get_latest_data_directory()}/statistics.json"
        statistics = load_statistics(stats_path)

        # Log the columns to verify that 'weekday' is present
        logging.info(f"Columns in dataset: {historical_data.columns}")

        # Train the overall model
        logging.info("Training overall prediction model...")
        overall_model = train_model(historical_data)
        save_model(overall_model, 'gradient_boosting_model_overall')

        logging.info("Model training and saving completed.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
