import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

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

def save_dataframe_to_latest_directory(df, filename):
    try:
        directory_path = get_latest_data_directory()
        file_path = os.path.join(directory_path, filename)
        df.to_csv(file_path, index=False)
        logging.info(f"DataFrame saved to {file_path}")
    except Exception as e:
        logging.error(f"Error in save_dataframe_to_latest_directory: {e}")
        raise

def save_dict_to_latest_directory(data, filename):
    try:
        directory_path = get_latest_data_directory()
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"JSON data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error in save_dict_to_latest_directory: {e}")
        raise

def load_latest_dataset():
    try:
        directory_path = get_latest_data_directory()
        dataset_path = os.path.join(directory_path, 'org_df.csv')
        df = pd.read_csv(dataset_path)
        logging.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error in load_latest_dataset: {e}")
        raise

def main():
    df = load_latest_dataset()
    df['draw_date'] = pd.to_datetime(df['draw_date'])
    last_draw = df.sort_values(by='draw_date', ascending=False).iloc[0]
    
    last_draw_day = last_draw['draw_date'].dayofweek
    last_draw_d = last_draw['d']
    
    if last_draw_day == 4:  # Friday
        next_tuesday_d = last_draw_d + 4
        next_friday_d = last_draw_d + 7
    else:  # Tuesday
        next_tuesday_d = last_draw_d + 3
        next_friday_d = last_draw_d + 6
    
    df['weekday'] = LabelEncoder().fit_transform(df['weekday'])
    
    X = df[['d', 'weekday']]
    y = df['total_sum']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logging.info(f"RMSE: {rmse}")

    next_tuesday_pred = model.predict(pd.DataFrame({'d': [next_tuesday_d], 'weekday': [1]}))
    next_friday_pred = model.predict(pd.DataFrame({'d': [next_friday_d], 'weekday': [0]}))

    predictions = {
        "next_tuesday": {
            "d": int(next_tuesday_d),
            "predicted_total_sum": float(next_tuesday_pred[0])
        },
        "next_friday": {
            "d": int(next_friday_d),
            "predicted_total_sum": float(next_friday_pred[0])
        }
    }
    
    # Saving predictions to target_draw_prediction.json
    target_file_path = os.path.join(get_latest_data_directory(), 'target_draw_prediction.json')
    with open(target_file_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    
    logging.info(f"Predictions saved to {target_file_path}")

if __name__ == "__main__":
    main()


