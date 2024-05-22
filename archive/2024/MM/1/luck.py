import pandas as pd
import itertools
import os
import json
import logging

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

def load_latest_dataset():
    try:
        base_path = 'data/prep'
        latest_date = sorted(os.listdir(base_path), reverse=True)[0]
        latest_time = sorted(os.listdir(os.path.join(base_path, latest_date)), reverse=True)[0]
        dataset_path = os.path.join(base_path, latest_date, latest_time, 'predicted_combinations.csv')
        return pd.read_csv(dataset_path)
    except Exception as e:
        logging.error(f"Error in load_latest_dataset: {e}")
        raise

def unique_combinations(df, n=25):
    # We will collect indices of rows in this list to ensure no duplicates
    selected_indices = set()
    unique_sets = []

    # Extract all unique mega_ball numbers
    unique_mega_balls = df['mega_ball'].unique()
    
    # We will iterate over unique mega_balls and for each one, we will try to find a unique set
    for mega_ball in unique_mega_balls:
        # Attempt to find a unique set with this mega_ball
        possible_rows = df[df['mega_ball'] == mega_ball]
        for _, row in possible_rows.iterrows():
            if row.name not in selected_indices:
                # Check if this row has any number already in the unique_sets
                if not any(row['w1'] in s for s in unique_sets) and \
                   not any(row['w2'] in s for s in unique_sets) and \
                   not any(row['w3'] in s for s in unique_sets) and \
                   not any(row['w4'] in s for s in unique_sets) and \
                   not any(row['w5'] in s for s in unique_sets):
                    # This row is unique so far, add it to unique_sets
                    unique_sets.append((row['w1'], row['w2'], row['w3'], row['w4'], row['w5'], mega_ball))
                    selected_indices.add(row.name)
                    break  # Break since we found our unique row for this mega_ball
        # Check if we have already found n unique sets
        if len(unique_sets) == n:
            break

    return unique_sets

def main():
    try:
        logging.info("Loading the latest datasets for Tuesday and Friday...")
        
        # Adjust these paths to match where your files are saved
        tuesday_path = os.path.join(get_latest_data_directory(), 'sorted_categorized_predicted_combinations_tuesday.csv')
        friday_path = os.path.join(get_latest_data_directory(), 'sorted_categorized_predicted_combinations_friday.csv')
        
        tuesday_df = pd.read_csv(tuesday_path)
        friday_df = pd.read_csv(friday_path)

        logging.info("Finding unique combinations for Tuesday...")
        unique_sets_tuesday = unique_combinations(tuesday_df)
        
        logging.info("Finding unique combinations for Friday...")
        unique_sets_friday = unique_combinations(friday_df)

        # Convert the unique sets into DataFrames
        unique_sets_df_tuesday = pd.DataFrame(unique_sets_tuesday, columns=['w1', 'w2', 'w3', 'w4', 'w5', 'mega_ball'])
        unique_sets_df_friday = pd.DataFrame(unique_sets_friday, columns=['w1', 'w2', 'w3', 'w4', 'w5', 'mega_ball'])

        logging.info("Saving the unique combinations to CSV...")
        save_dataframe_to_latest_directory(unique_sets_df_tuesday, 'unique_combinations_tuesday.csv')
        save_dataframe_to_latest_directory(unique_sets_df_friday, 'unique_combinations_friday.csv')

        logging.info("Process completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
