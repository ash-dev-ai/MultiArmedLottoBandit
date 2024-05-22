from datetime import datetime
from prep import fetch_and_preprocess
import pandas as pd
import os

df_steps = None
df_steps_i = None

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
    global df_steps, df_steps_i  
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
        for col in ['w1', 'w2', 'w3', 'w4', 'w5', 'pb']:
            diff_dict[f'dw{col[1:]}'] = df.iloc[i][col] - df.iloc[i-1][col]

        # Within-day differences between adjacent 'w*' and 'pb'
        for j in range(1, 5):
            diff_dict[f'rw{j}-{j+1}'] = df.iloc[i][f'w{j}'] - df.iloc[i][f'w{j+1}']
        diff_dict['rwpb'] = df.iloc[i]['w5'] - df.iloc[i]['pb']

        # Append calculated differences to df_steps
        for key, value in diff_dict.items():
            df_steps.at[i, key] = value

    # Step 5: Create DataFrame for "imaginary" results
    df_steps_i = df_steps.copy()
    for col in df_steps.columns:
        if pd.api.types.is_datetime64_any_dtype(df_steps[col]):
            df_steps_i[col] = df_steps[col]
        elif isinstance(df_steps[col].iloc[0], list):
            df_steps_i[col] = df_steps[col]
        else:
            df_steps_i[col] = -df_steps[col]

    # Export DataFrames to CSV with date and time in the file names
    df_steps.to_csv(f'{data_path}/df_steps.csv', index=False)
    df_steps_i.to_csv(f'{data_path}/df_steps_i.csv', index=False)            
    
    # Log the end time
    end_time = datetime.now()
    print(f"Script ended at: {end_time}")
    print(f"Total time taken: {end_time - start_time}")

    return df_steps, df_steps_i

# For demonstration, calling the function and showing the first few rows
if __name__ == '__main__':
    df_steps, df_steps_i = calculate_steps()
    print("Real steps:")
    print(df_steps.head())
    print("Imaginary steps:")
    print(df_steps_i.head())
