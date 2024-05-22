# prep_pb.py
import pandas as pd
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_pb_data(df):
    """Prepares the Powerball (PB) data for analysis, handling ISO 8601 date format."""
    logging.info("Preparing Powerball data...")
    try:
        # Rename 'Winning' column to 'winning_numbers'
        df.rename(columns={'Winning': 'winning_numbers'}, inplace=True)

        # Convert 'draw_date' to datetime format (handling ISO 8601)
        df['draw_date'] = pd.to_datetime(df['draw_date'], format='ISO8601')

        # Find the earliest date in the dataset
        earliest_date = df['draw_date'].min()

        # Create a new column 'days_since_earliest' 
        df['days_since_earliest'] = (df['draw_date'] - earliest_date).dt.days

        # Remove spaces and separate the winning numbers
        df['winning_numbers'] = (
            df['winning_numbers']
            .astype(str)
            .str.replace(" ", "", regex=False)
        )

        # Convert to list of ints, handling potential errors (e.g., non-numeric characters)
        def convert_to_int_list(x):
            try:
                return list(map(int, x.split()))
            except ValueError:
                logging.warning(f"Invalid winning numbers string: {x}")
                return []  # Or some other default handling

        df['winning_numbers'] = df['winning_numbers'].apply(convert_to_int_list)

        # Calculate the sum of winning numbers
        df['winning_numbers_sum'] = df['winning_numbers'].apply(sum)
        # Calculate the sum of winning numbers and Powerball
        df['sum_and_pb'] = df['winning_numbers'].apply(lambda x: sum(x))

        # Create new columns 'w1' to 'w5' and 'pb' for winning numbers and Powerball
        for i in range(1, 6):
            df[f'w{i}'] = df['winning_numbers'].apply(lambda x: x[i-1] if len(x) >= i else None)
        df['pb'] = df['winning_numbers'].apply(lambda x: x[-1] if x else None)

        # Drop the 'Multiplier' column
        df.drop(columns=['Multiplier'], inplace=True, errors='ignore')

        # Add weekday column (using correct method for filtering on datetime)
        df['weekday'] = pd.DatetimeIndex(df['draw_date']).day_name()

        # Filter out draws not on Monday, Wednesday, or Saturday
        df = df[df['weekday'].isin(['Monday', 'Wednesday', 'Saturday'])]

        logging.info(f"Shape of prepared_pb_data: {df.shape}")  # Log final shape
        logging.info("Powerball data prepared successfully.")
        return df

    except Exception as e:
        logging.error(f"Failed to prepare PB data: {e}")
        raise