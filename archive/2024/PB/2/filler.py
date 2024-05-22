import pandas as pd
import random
import os
from datetime import datetime
import prep

# Create directories for saving data and visuals
current_date = datetime.now().strftime("%Y%m%d")
current_time = datetime.now().strftime('%H%M%S')

# Create a folder named 'data/filler' if it doesn't exist
data_path = f'data/filler/{current_date}/{current_time}'
if not os.path.exists('data/filler'):
    os.makedirs('data/filler')
os.makedirs(data_path)

# Create a folder named 'visuals/filler' if it doesn't exist
visuals_path = f'visuals/filler/{current_date}/{current_time}'
if not os.path.exists('visuals/filler'):
    os.makedirs('visuals/filler')
os.makedirs(visuals_path)

def generate_random_numbers():
    '''Generate random winning numbers and a Powerball number'''
    winning_numbers = random.sample(range(1, 70), 5)  # 5 unique numbers from 1 to 69
    powerball = random.randint(1, 26)  # Powerball number from 1 to 26
    return winning_numbers, powerball


def preprocess_data(df, earliest_date):
    # Ensure 'draw_date' is in datetime format
    df['draw_date'] = pd.to_datetime(df['draw_date'], errors='coerce')

    # Remove rows with NaN in 'draw_date' after conversion
    df = df.dropna(subset=['draw_date'])

    # Use the provided 'earliest_date' for calculation
    # Convert 'earliest_date' to datetime if it's not already
    earliest_date = pd.to_datetime(earliest_date)

    # Create a new column 'd' containing the number of days since the earliest date from org_df
    df['d'] = (df['draw_date'] - earliest_date).dt.days

    # Check if 'winning_numbers' is a string and needs splitting, then extend the list with the Powerball number
    if df['winning_numbers'].dtype == object:
        df['winning_numbers'] = df['winning_numbers'].apply(lambda x: list(map(int, x.split())) if isinstance(x, str) else x)

    # Create new columns 'w1' to 'w5' for the first 5 winning numbers
    for i in range(1, 6):
        df[f'w{i}'] = df['winning_numbers'].apply(lambda x: x[i-1] if len(x) > i-1 else None)

    # Create new column 'pb' for the Powerball number, ensuring there are enough elements in the list
    df['pb'] = df['winning_numbers'].apply(lambda x: x[5] if len(x) > 5 else None)

    # Drop 'multiplier' column if it exists
    df.drop(columns=['multiplier'], inplace=True, errors='ignore')

    return df

def calculate_steps(df):
    df.sort_values('d', inplace=True)
    df_steps = df.copy()

    for i in range(1, df.shape[0]):
        diff_dict = {}

        # Day-to-day differences for 'w*' and 'pb'
        diff_dict['dw1'] = df.iloc[i]['w1'] if i == 0 else df.iloc[i]['w1'] - df.iloc[i-1]['w1']
        for col in ['w2', 'w3', 'w4', 'w5', 'pb']:
            diff_dict[f'dw{col[1:]}'] = df.iloc[i][col] - df.iloc[i-1][col]

        # Within-day differences between 'w*' and 'pb'
        diff_dict['rw1-2'] = df.iloc[i]['w1']
        for j in range(2, 5):
            diff_dict[f'rw{j}-{j+1}'] = df.iloc[i][f'w{j}'] - df.iloc[i][f'w{j+1}']
        diff_dict['rwpb'] = df.iloc[i]['w5'] - df.iloc[i]['pb']

        for key, value in diff_dict.items():
            df_steps.at[i, key] = value

    return df_steps

def fill_missing_dates(org_df, output_csv_path):
    # Find the earliest date in org_df for 'd' calculation
    earliest_date = org_df['draw_date'].min()

    # Process org_df with the earliest_date
    org_df = preprocess_data(org_df, earliest_date)

    # Create a complete date range from the earliest date in org_df to the current date
    latest_date = datetime.now().strftime('%Y-%m-%d')  # Use current date as the latest date
    all_dates = pd.date_range(start=earliest_date, end=latest_date).to_pydatetime().tolist()

    # Convert dates in org_df to datetime for comparison
    org_df['draw_date'] = pd.to_datetime(org_df['draw_date'])

    # Get a set of existing dates in org_df for efficient lookup
    existing_dates = set(org_df['draw_date'])

    # Initialize new_df as a copy of org_df
    new_df = org_df.copy()

    # Loop through each date in the range and fill missing dates with random data
    for date in all_dates:
        if date not in existing_dates:
            winning_numbers, powerball = generate_random_numbers()
            new_entry = {
                'draw_date': date,
                'winning_numbers': ' '.join(map(str, winning_numbers)) + ' ' + str(powerball),
                'd': None,  # 'd' will be calculated later
                'w1': winning_numbers[0],
                'w2': winning_numbers[1],
                'w3': winning_numbers[2],
                'w4': winning_numbers[3],
                'w5': winning_numbers[4],
                'pb': powerball
            }
            new_df = new_df.append(new_entry, ignore_index=True)

    # Process new_df with the earliest_date
    new_df = preprocess_data(new_df, earliest_date)

    # Sort the DataFrame by 'draw_date'
    new_df.sort_values(by='draw_date', inplace=True)

    # Save new_df
    new_df.to_csv(output_csv_path, index=False)

    return new_df

if __name__ == '__main__':
    org_df = prep.fetch_and_preprocess()  # Use the org_df from prep.py
    output_csv_path = os.path.join(data_path, 'new_df.csv')
    filled_df = fill_missing_dates(org_df, output_csv_path)
    
    # Now, calculate the steps for filled_df
    df_steps = calculate_steps(filled_df)

    # Save df_steps to a CSV file in the data_path
    steps_output_path = os.path.join(data_path, 'filled_df_steps.csv')
    df_steps.to_csv(steps_output_path, index=False)

    print("New DataFrame saved to:", output_csv_path)
    print("Filled df_steps DataFrame saved to:", steps_output_path)
    print(filled_df.head())
    print(df_steps.head())