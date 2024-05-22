# save.py
import os
import sqlite3
import pandas as pd
import json
import numpy as np
import datetime
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_data_to_csv(data, file_path):
    """Saves DataFrame to CSV."""
    logging.info(f"Saving data to CSV: {file_path}")
    try:
        data.to_csv(file_path, index=False)
    except Exception as e:
        logging.error(f"Error saving data to CSV: {e}")

def save_data_to_database(data, conn, table_name, batch_size=250000):
    """Saves DataFrame to database table."""
    logging.info(f"Saving data to database table: {table_name}")
    try:
        # Ensure correct column order and convert types as needed
        if table_name == 'prepared_mb_data':
            correct_column_order = ['draw_date', 'winning_numbers', 'mega_ball', 'days_since_earliest', 
                                    'winning_numbers_sum', 'total_sum', 'sum_and_mega_ball', 'w1', 'w2', 'w3', 'w4', 'w5', 'weekday']
        elif table_name == 'prepared_pb_data':
            correct_column_order = ['draw_date', 'winning_numbers', 'pb', 'days_since_earliest', 
                                    'winning_numbers_sum', 'w1', 'w2', 'w3', 'w4', 'w5', 'weekday']  # Removed 'sum_and_pb'
        else:
            raise ValueError(f"Unknown table name: {table_name}")
        
        # Reorder columns and convert to appropriate types
        data = data[correct_column_order].copy()

        data['draw_date'] = pd.to_datetime(data['draw_date']).dt.strftime('%Y-%m-%d')
        data['winning_numbers'] = data['winning_numbers'].apply(lambda x: json.dumps(x, cls=NumpyEncoder)) 
        data = data.astype(str)
        
        # Batch insert to Database
        cursor = conn.cursor()

        # Batch insertion for efficiency
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i : i + batch_size]
            placeholders = ', '.join(['?'] * len(batch.columns))
            columns = ', '.join(batch.columns)
            cursor.executemany(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", batch.values)
        
        conn.commit()
    except Exception as e:
        logging.error(f"Error saving data to database: {e}")
        raise

def save_statistics(stats, file_path):
    """Saves statistical analysis results to a JSON file."""
    try:
        # Convert DataFrame to dictionary if needed
        if isinstance(stats, pd.DataFrame):
            stats = stats.astype(str).to_dict(orient='list') # Convert to string

        with open(file_path, 'w') as f:
            json.dump(stats, f, indent=4)
        logging.info(f"Statistics saved to {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving statistics to JSON: {e}")

def create_table_if_not_exists(conn, table_name, schema):
    """
    Creates a table in the database if it doesn't already exist, based on the provided schema.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} {schema}")
        conn.commit()
        logging.info(f"Table {table_name} is ready in the database.")
    except Exception as e:
        logging.error(f"Error occurred while creating table {table_name}: {e}")

def insert_data_to_table(conn, table_name, data, batch_size=10000):
    """
    Inserts data into a specific table in the database in batches.
    """
    try:
        cursor = conn.cursor()
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            placeholders = ', '.join(['?'] * len(batch[0]))
            cursor.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", batch)
        conn.commit()
        logging.info(f"Data inserted into {table_name} in the database.")
    except Exception as e:
        logging.error(f"Error occurred while inserting data into table {table_name}: {e}")
