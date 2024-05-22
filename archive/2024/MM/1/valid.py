import pandas as pd
import json
import os
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
        
def load_target_draw_prediction():
    directory_path = get_latest_data_directory()
    file_path = os.path.join(directory_path, 'target_draw_prediction.json')
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Round the predicted_total_sum to the nearest whole number for both draws
    rounded_predictions = {
        'next_tuesday': round(data['next_tuesday']['predicted_total_sum']),
        'next_friday': round(data['next_friday']['predicted_total_sum'])
    }
    return rounded_predictions

def categorize_difference(row, predicted_total_sum):
    if row['total_sum'] == predicted_total_sum:
        return 'Match'
    difference = abs(row['total_sum'] - predicted_total_sum)
    if difference <= 50:
        return 'Very Close'
    elif difference <= 100:
        return 'Close'
    elif difference <= 150:
        return 'Moderate'
    else:
        return 'Far'

def main():
    try:
        logging.info("Loading datasets...")
        valid_combinations_df = load_latest_dataset()
        predictions = load_target_draw_prediction()
        
        # Apply the categorization for both Tuesday and Friday
        valid_combinations_df['Tuesday_Category'] = valid_combinations_df.apply(
            categorize_difference, args=(predictions['next_tuesday'],), axis=1)
        valid_combinations_df['Friday_Category'] = valid_combinations_df.apply(
            categorize_difference, args=(predictions['next_friday'],), axis=1)
        
        # Assuming you want to sort and save based on one of the predictions; you may adjust as needed
        # For a more sophisticated sort considering both predictions, additional logic will be required
        category_order = {'Match': 0, 'Very Close': 1, 'Close': 2, 'Moderate': 3, 'Far': 4}
        valid_combinations_df['Sort_Key_Tuesday'] = valid_combinations_df['Tuesday_Category'].map(category_order)
        sorted_df_tuesday = valid_combinations_df.sort_values(by='Sort_Key_Tuesday').drop(['Sort_Key_Tuesday', 'Friday_Category'], axis=1)
        
        valid_combinations_df['Sort_Key_Friday'] = valid_combinations_df['Friday_Category'].map(category_order)
        sorted_df_friday = valid_combinations_df.sort_values(by='Sort_Key_Friday').drop(['Sort_Key_Friday', 'Tuesday_Category'], axis=1)
        
        # Save the sorted DataFrames for Tuesday and Friday
        save_dataframe_to_latest_directory(sorted_df_tuesday, 'sorted_categorized_predicted_combinations_tuesday.csv')
        save_dataframe_to_latest_directory(sorted_df_friday, 'sorted_categorized_predicted_combinations_friday.csv')
        
        logging.info("Process completed and all outputs saved.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()