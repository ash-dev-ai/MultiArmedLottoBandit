# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:44:01 2024

@author: AVMal
"""

import pandas as pd
from datetime import datetime
from prep import fetch_and_preprocess
import os

# Declare df_steps as a global variable
global df_steps
df_steps = None

# Create directories for saving data and visuals
current_date = datetime.now().strftime("%Y%m%d")
current_time = datetime.now().strftime('%H%M%S')

# Create a folder named 'data' if it doesn't exist
data_path = f'data/steps/{current_date}/{current_time}'
if not os.path.exists('data/steps'):
    os.makedirs('data/steps')
os.makedirs(data_path)

# Create a folder named 'visuals/steps' if it doesn't exist
visuals_path = f'visuals/steps/{current_date}/{current_time}'
if not os.path.exists('visuals/steps'):
    os.makedirs('visuals/steps')
os.makedirs(visuals_path)

def calculate_steps():
    # Access the global df_steps variable
    global df_steps
    
    # Log the start time
    start_time = datetime.now()
    print(f"Script started at: {start_time}")

    # Step 1: Import DataFrame from prep.py
    df = fetch_and_preprocess()

    # Step 2: Sort by 'd'
    df.sort_values('d', inplace=True)

    # Step 3: Copy the full DataFrame from df
    df_steps = df.copy()

    # Step 4: Loop through sorted DataFrame to calculate additional columns and steps
    for i in range(1, df.shape[0]):
        diff_dict = {}  # Dictionary to store differences

        # Day-to-day differences for each 'w*' and 'pb'
        diff_dict['dw1'] = df.iloc[i]['w1'] - df.iloc[i-1]['w1']
        for col in ['w2', 'w3', 'w4', 'w5', 'pb']:
            diff_dict[f'dw{col[1:]}'] = pd.to_numeric(df.iloc[i][col]) - pd.to_numeric(df.iloc[i-1][col])

        # Within-day differences between adjacent 'w*' and 'pb'
        diff_dict['rw1-2'] = pd.to_numeric(df.iloc[i]['w1'])
        for j in range(2, 5):
            diff_dict[f'rw{j}-{j+1}'] = pd.to_numeric(df.iloc[i][f'w{j}']) - pd.to_numeric(df.iloc[i][f'w{j+1}'])
        diff_dict['rwpb'] = pd.to_numeric(df.iloc[i]['w5']) - pd.to_numeric(df.iloc[i]['pb'])

        # Append calculated differences to df_steps
        for key, value in diff_dict.items():
            df_steps.at[i, key] = value

    # Export DataFrames to CSV with date and time in the file names
    df_steps.to_csv(f'{data_path}/df_steps.csv', index=False)
    
    # Log the end time
    end_time = datetime.now()
    print(f"Script ended at: {end_time}")
    print(f"Total time taken: {end_time - start_time}")

if __name__ == '__main__':
    # Call the calculate_steps() function to perform data processing
    calculate_steps()
    
    # Now, you can print the first few rows of df_steps
    if df_steps is not None:
        print("Real steps:")
        print(df_steps.head())
    else:
        print("df_steps is None. Check the calculate_steps() function.")
