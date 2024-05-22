# main.py
import sqlite3
import logging
# Import API settings from settings.py
from config.settings import API_ENDPOINT, API_LIMIT  

# Import modules with corrected paths
from load import load_org_mb_data, load_org_pb_data  
from prep_pb import prepare_pb_data
from prep_mb import prepare_mb_data
from stats import extract_statistical_measures, extract_date_features, extract_historical_patterns, extract_combination_characteristics
from combinations_all import CombinationProcessor  # Import the class
from combinations_mb import CombinationProcessor as CombinationProcessorMB
from combinations_pb import CombinationProcessor as CombinationProcessorPB
from save import save_data_to_database, save_statistics, create_table_if_not_exists
from stacked_lstm import main as train_lstm_model


def main():
    # Set file paths
    db_file_path = "./data/lottery_data.db" 
    mb_stats_file_path = "./data/mb_stats.json"
    pb_stats_file_path = "./data/pb_stats.json"

    try:
        # Database connection setup using context manager
        with sqlite3.connect(db_file_path) as conn:
            # Ensure tables exist in the database (updated schema to include sum_and_mega_ball for 'prepared_mb_data')
            create_table_if_not_exists(conn, 'prepared_mb_data', '(draw_date TEXT, winning_numbers TEXT, mega_ball INTEGER, winning_numbers_sum INTEGER, total_sum INTEGER, w1 INTEGER, w2 INTEGER, w3 INTEGER, w4 INTEGER, w5 INTEGER, days_since_earliest INTEGER, weekday TEXT, sum_and_mega_ball TEXT)')  
            create_table_if_not_exists(conn, 'prepared_pb_data', '(draw_date TEXT, winning_numbers TEXT, pb INTEGER, winning_numbers_sum INTEGER, sum_and_pb TEXT, w1 INTEGER, w2 INTEGER, w3 INTEGER, w4 INTEGER, w5 INTEGER, days_since_earliest INTEGER, weekday TEXT)')
        
            org_mb_data = load_org_mb_data(API_ENDPOINT, API_LIMIT)
            org_pb_data = load_org_pb_data(API_ENDPOINT, API_LIMIT)
    
            logging.info(f"Shape of org_mb_data: {org_mb_data.shape}") # Log initial shape of dataframe
            logging.info(f"Shape of org_pb_data: {org_pb_data.shape}")
    
            prepared_mb_data = prepare_mb_data(org_mb_data.copy()) # Create a copy of the original DataFrame
            prepared_pb_data = prepare_pb_data(org_pb_data.copy())
    
            logging.info(f"Shape of prepared_mb_data: {prepared_mb_data.shape}") # Log shape after prep
            logging.info(f"Shape of prepared_pb_data: {prepared_pb_data.shape}") 

            # Extract date features
            extract_date_features(prepared_mb_data)
            extract_date_features(prepared_pb_data)
            
            # Extract historical pattern features
            extract_historical_patterns(prepared_mb_data)
            extract_historical_patterns(prepared_pb_data)
            
            # Extract combination characteristics
            extract_combination_characteristics(prepared_mb_data)
            extract_combination_characteristics(prepared_pb_data)

            # Save statistics (modify to save the extracted features)
            save_statistics(prepared_mb_data, mb_stats_file_path)
            save_statistics(prepared_pb_data, pb_stats_file_path)

            # Save prepared data to the database
            save_data_to_database(prepared_mb_data, conn, 'prepared_mb_data')
            save_data_to_database(prepared_pb_data, conn, 'prepared_pb_data')

            # Generate all combinations and update validity for MB and PB
            CombinationProcessor.main()  
            CombinationProcessorMB.main() 
            CombinationProcessorPB.main()

            # Train LSTM model
            train_lstm_model()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

