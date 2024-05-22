import pandas as pd
import os
import logging
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tcn import TCN, tcn_full_summary

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

def load_statistics(stats_path):
    try:
        with open(stats_path, 'r') as f:
            statistics = json.load(f)
        return statistics
    except Exception as e:
        logging.error(f"Error in load_statistics: {e}")
        raise

def load_csv(filename):
    try:
        directory_path = get_latest_data_directory()
        file_path = os.path.join(directory_path, filename)
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error in load_csv: {e}")
        raise

def load_tcn_model():
    try:
        directory_path = get_latest_data_directory()
        model_path = os.path.join(directory_path, 'gradient_boosting_model_overall.h5')
        with tf.keras.utils.custom_object_scope({'TCN': TCN}):
            model = tf.keras.models.load_model(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error in load_tcn_model: {e}")
        raise

def prepare_data_for_prediction(df):
    # Adjust based on the exact features used during model training in org.py
    # This assumes the model expects data in a specific shape, adjust as necessary
    df['winning_numbers_sum'] = df[['w1', 'w2', 'w3', 'w4', 'w5']].sum(axis=1)
    df['total_sum'] = df['winning_numbers_sum'] + df['mega_ball']
    X = df[['winning_numbers_sum', 'w1', 'w2', 'w3', 'w4', 'w5', 'mega_ball', 'total_sum']].values
    # Reshape the data if your model expects a specific input shape, e.g., for TCN models
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    return X

def predict_model(model, X):
    try:
        predictions = model.predict(X)
        logging.info("Predictions completed.")
        return predictions.flatten()  # Flatten predictions if necessary
    except Exception as e:
        logging.error(f"Error in predict_model: {e}")
        raise

def save_predictions(df, predictions, filename):
    try:
        # Assuming 'predictions' is a 1D NumPy array and 'df' is your original DataFrame
        df['Predictions'] = predictions  # Add predictions as a new column
        
        directory_path = get_latest_data_directory()
        file_path = os.path.join(directory_path, filename)
        df.to_csv(file_path, index=False)
        logging.info(f"DataFrame with predictions saved to {file_path}")
    except Exception as e:
        logging.error(f"Error in save_predictions: {e}")
        raise

def main():
    try:
        model = load_tcn_model()  # Assuming this loads your model correctly
        valid_combinations = load_csv('valid_predictions.csv')
        X = prepare_data_for_prediction(valid_combinations)
        predictions = predict_model(model, X)
        
        # Assuming 'valid_combinations' is the DataFrame you want to include in the output
        save_predictions(valid_combinations, predictions, 'predicted_combinations.csv')
    except Exception as e:
        logging.error(f"Main function error: {e}")
        raise

        
if __name__ == '__main__':
    main()