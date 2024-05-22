# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:43:06 2024

@author: AVMal
"""

import pandas as pd
from sodapy import Socrata
import warnings
from datetime import datetime
import os

# Initialize org_df as an empty DataFrame initially
org_df = pd.DataFrame()

# Create directories for saving data and visuals
current_date = datetime.now().strftime("%Y%m%d")
current_time = datetime.now().strftime('%H%M%S')

# Create a folder named 'data' if it doesn't exist
data_path = f'data/prep/{current_date}/{current_time}'
if not os.path.exists('data/prep'):
    os.makedirs('data/prep')
os.makedirs(data_path)

# Create a folder named 'visuals/study000' if it doesn't exist
visuals_path = f'visuals/prep/{current_date}/{current_time}'
if not os.path.exists('visuals/prep'):
    os.makedirs('visuals/prep')
os.makedirs(visuals_path)

def fetch_and_preprocess():
    global org_df  # Access the global variable org_df
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Log the start time
    start_time = datetime.now()
    print(f"Script started at: {start_time}")

    # Fetch data from the API and convert to DataFrame
    client = Socrata("data.ny.gov", None)
    results = client.get("5xaw-6ayf", limit=5000)
    df = pd.DataFrame.from_records(results)

    # Convert 'draw_date' to datetime format
    df['draw_date'] = pd.to_datetime(df['draw_date'])

    # Find the earliest date in the dataset
    earliest_date = df['draw_date'].min()

    # Create a new column 'd' containing the number of days since the earliest date
    df['d'] = (df['draw_date'] - earliest_date).dt.days

    # Split the 'winning_numbers' into a list of integers
    df['winning_numbers'] = df['winning_numbers'].apply(lambda x: list(map(int, x.split())))

    # Create new columns 'w1' to 'w5' for the first 5 winning numbers
    for i in range(1, 6):
        df[f'w{i}'] = df['winning_numbers'].apply(lambda x: x[i-1])

    # Create new column 'pb' for the 6th winning number (Powerball)
    df.rename(columns={'mega_ball': 'pb'}, inplace=True)

    # Drop 'multiplier' column
    df.drop(columns=['multiplier'], inplace=True)

    # Set org_df as the global DataFrame
    org_df = df

    # Export DataFrame to CSV with date and time in the file name
    org_df.to_csv(f'{data_path}/org_df.csv', index=False)
    
    # Log the end time
    end_time = datetime.now()
    print(f"Script ended at: {end_time}")
    print(f"Total time taken: {end_time - start_time}")

    return df

# Call the fetch_and_preprocess function to initialize org_df
fetch_and_preprocess()