# prep_mb.py
import pandas as pd
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_mb_data(df):
    """
    Prepares the Mega Ball (MB) data for analysis.
    """
    try:
        df['draw_date'] = pd.to_datetime(df['draw_date'])
        earliest_date = df['draw_date'].min()

        # Create days_since_earliest column
        df['days_since_earliest'] = (df['draw_date'] - earliest_date).dt.days  

        df['mega_ball'] = pd.to_numeric(df['mega_ball'], errors='coerce')

        # Split the 'winning_numbers' into a list of integers
        df['winning_numbers'] = df['winning_numbers'].apply(lambda x: list(map(int, x.split())))

        # Calculate winning_numbers_sum BEFORE converting to a string
        df['winning_numbers_sum'] = df['winning_numbers'].apply(sum)
        df['total_sum'] = df['winning_numbers_sum'] + df['mega_ball']
        
        # Convert 'winning_numbers' to a string representation of a list
        df['winning_numbers'] = df['winning_numbers'].apply(lambda x: ','.join(map(str, x))) 

        df['sum_and_mega_ball'] = df.apply(lambda row: [row['winning_numbers_sum'], row['mega_ball']], axis=1)

        # Create new columns 'w1' to 'w5' for the first 5 winning numbers
        for i in range(1, 6):
            df[f'w{i}'] = df['winning_numbers'].apply(lambda x: int(x.split(',')[i-1]))

        # Drop 'multiplier' column, if present
        df.drop(columns=['multiplier'], inplace=True, errors='ignore')

        df['weekday'] = df['draw_date'].dt.day_name()
        df['weekday'] = df['weekday'].apply(lambda x: x if x in ['Tuesday', 'Friday'] else None)

        # Debugging statements to print column names and data types
        logging.info("Column Names: %s", df.columns)
        logging.info("Data Types: %s", df.dtypes)

        return df

    except Exception as e:
        logging.error(f"Failed to prepare MB data: {e}")
        raise

