# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:43:06 2024

@author: AVMal
"""
import os
import json
import math
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
from sodapy import Socrata
import itertools
import sqlite3

# Constants
API_ENDPOINT = "data.ny.gov"
API_LIMIT = 5000

# Initialize org_df as an empty DataFrame initially
org_df = pd.DataFrame()

# Create directories for saving data and visuals
current_date = datetime.now().strftime("%Y%m%d")
current_time = datetime.now().strftime('%H%M%S')

data_path = f'data/prep/{current_date}/{current_time}'
os.makedirs(data_path, exist_ok=True)

visuals_path = f'visuals/prep/{current_date}/{current_time}'
os.makedirs(visuals_path, exist_ok=True)

def fetch_and_preprocess():
    global org_df 
    warnings.filterwarnings("ignore")

    start_time = datetime.now()
    print(f"Script started at: {start_time}")

    # Fetch data from the API
    client = Socrata(API_ENDPOINT, None)
    results = client.get("5xaw-6ayf", limit=API_LIMIT)
    df = pd.DataFrame.from_records(results)

    df['draw_date'] = pd.to_datetime(df['draw_date'])
    earliest_date = df['draw_date'].min()
    df['d'] = (df['draw_date'] - earliest_date).dt.days

    df['mega_ball'] = pd.to_numeric(df['mega_ball'], errors='coerce')
    df['winning_numbers'] = df['winning_numbers'].apply(lambda x: list(map(int, x.split())))
    df['winning_numbers_sum'] = df['winning_numbers'].apply(sum)
    
    df['total_sum'] = df['winning_numbers_sum'] + df['mega_ball']

    df['Sum and Mega Ball'] = df.apply(lambda row: f"[{row['winning_numbers_sum']}, {row['mega_ball']}]", axis=1)
    for i in range(1, 6):
        df[f'w{i}'] = df['winning_numbers'].apply(lambda x: x[i-1])
    df.drop(columns=['multiplier'], inplace=True)

    df['weekday'] = df['draw_date'].dt.day_name()
    df['weekday'] = df['weekday'].apply(lambda x: x if x in ['Tuesday', 'Friday'] else None)

    org_df = df
    org_df.to_csv(f'{data_path}/org_df.csv', index=False)

    end_time = datetime.now()
    print(f"Script ended at: {end_time}")
    print(f"Total time taken: {end_time - start_time}")

    return df

def statistical_analysis(df):
    stats = {
        'winning_numbers_sum': {
            'Mean': df['winning_numbers_sum'].mean(),
            'Median': df['winning_numbers_sum'].median(),
            'Mode': df['winning_numbers_sum'].mode()[0], 
            'Standard Deviation': df['winning_numbers_sum'].std()
        },
        'mega_ball': {
            'Mean': df['mega_ball'].mean(),
            'Median': df['mega_ball'].median(),
            'Mode': df['mega_ball'].mode()[0],
            'Standard Deviation': df['mega_ball'].std()
        }
    }

    for i in range(1, 6):
        col_name = f'w{i}'
        stats[col_name] = {
            'Mean': df[col_name].mean(),
            'Median': df[col_name].median(),
            'Mode': df[col_name].mode()[0],
            'Standard Deviation': df[col_name].std()
        }

    return stats

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def matches_historical(comb, historical_data):
    for entry in historical_data:
        if len(set(comb) & set(entry)) >= 3:
            return True
    return False

def has_consecutive_sequence(comb):
    sorted_comb = sorted(comb)
    consecutive_count = 1
    for i in range(1, len(sorted_comb)):
        if sorted_comb[i] == sorted_comb[i - 1] + 1:
            consecutive_count += 1
            if consecutive_count >= 3:
                return True
        else:
            consecutive_count = 1
    return False

def combination_generator(n, k):
    """Generate combinations of n choose k."""
    for comb in itertools.combinations(range(1, n+1), k):
        for pb in range(1, 26):  # Assuming pb is the Powerball or equivalent
            yield comb + (pb,)

def filter_and_export_valid_combinations(df, export_path):
    """
    Generate valid combinations based on historical data, apply necessary filters,
    and export valid combinations to a CSV file.
    """
    historical_data = df[['w1', 'w2', 'w3', 'w4', 'w5', 'mega_ball']].apply(tuple, axis=1).tolist()
    valid_combinations = []

    # Iterate through combinations and apply filters
    for comb in combination_generator(70, 5):  # Using the previously defined combination generator
        if not matches_historical(comb[:-1], historical_data) and not has_consecutive_sequence(comb[:-1]):
            valid_combinations.append(comb)

    # Convert valid combinations to a DataFrame and export to CSV
    valid_df = pd.DataFrame(valid_combinations, columns=['w1', 'w2', 'w3', 'w4', 'w5', 'mega_ball'])
    valid_df.to_csv(export_path, index=False)
    print(f"Valid combinations exported to {export_path}")

def create_combinations_table(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS combinations (
            w1 INTEGER,
            w2 INTEGER,
            w3 INTEGER,
            w4 INTEGER,
            w5 INTEGER,
            mega_ball INTEGER
        )
    """)
    conn.commit()

def insert_combinations_to_db(conn, combinations_generator, batch_size=10000):
    """Insert combinations into the database in batches."""
    cursor = conn.cursor()
    batch = []
    for comb in combinations_generator:
        batch.append(comb)
        if len(batch) >= batch_size:
            cursor.executemany("""
                INSERT INTO combinations (w1, w2, w3, w4, w5, mega_ball) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, batch)
            conn.commit()
            batch = []
    # Insert any remaining combinations in the batch
    if batch:
        cursor.executemany("""
            INSERT INTO combinations (w1, w2, w3, w4, w5, mega_ball) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, batch)
        conn.commit()

if __name__ == "__main__":
    # Assuming 'data.db' is the SQLite database file
    conn = sqlite3.connect('data.db')

    df = fetch_and_preprocess()
    stats = statistical_analysis(df)
    print(stats)

    # Define the path for exporting valid combinations
    valid_combinations_path = f"{data_path}/valid_predictions.csv"
    filter_and_export_valid_combinations(df, valid_combinations_path)

    # Save statistics to JSON with NumpyEncoder to handle Numpy data types
    stats_path = f"{data_path}/statistics.json"
    with open(stats_path, 'w') as stats_file:
        json.dump(stats, stats_file, indent=4, cls=NumpyEncoder)
    print(f"Statistical analysis saved to {stats_path}")

    conn.close()
