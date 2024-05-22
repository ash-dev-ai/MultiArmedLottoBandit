# stats.py
import pandas as pd
import numpy as np
import logging

def extract_date_features(df):
    """Extracts date-related features from the DataFrame."""
    logging.info("Extracting date features...")
    try:
        # Ensure 'draw_date' is in a consistent format
        df['draw_date'] = pd.to_datetime(df['draw_date'], dayfirst=True, errors='coerce')  

        # Handle missing dates
        df.dropna(subset=['draw_date'], inplace=True)  

        df['year'] = df['draw_date'].dt.year
        df['month'] = df['draw_date'].dt.month
        df['day'] = df['draw_date'].dt.day
        df['weekday'] = df['draw_date'].dt.dayofweek
        df['week'] = df['draw_date'].dt.isocalendar().week
        logging.info("Date features extracted successfully.")
        return df
    except Exception as e:
        logging.error(f"Error occurred during date feature extraction: {e}")
        return None  # Return None to signal failure

def extract_historical_patterns(df):
    """Extracts historical patterns from the DataFrame."""
    logging.info("Extracting historical patterns...")
    try:
        number_columns = [col for col in df.columns if col.startswith('w') or col == 'pb']
        number_counts = {col: {} for col in number_columns}

        for index, row in df.iterrows():
            for col in number_columns:
                num = row[col]
                # Convert num to string to make it hashable
                num_str = str(num)
                if not pd.isna(num): 
                    number_counts[col][num_str] = number_counts[col].get(num_str, 0) + 1
                    df.at[index, f'{col}_frequency'] = number_counts[col][num_str] - 1

        logging.info("Historical patterns extracted successfully.")
        return df
    except Exception as e:
        logging.error(f"Error occurred during historical pattern extraction: {e}")
        raise

def extract_statistical_measures(df):
    logging.info("Extracting statistical measures...")
    try:
        # Calculate for white balls only
        white_balls = df[['w1', 'w2', 'w3', 'w4', 'w5']].values.tolist()
        df['mean'] = [np.mean(x) if x else np.nan for x in white_balls]
        df['median'] = [np.median(x) if x else np.nan for x in white_balls]
        df['std_deviation'] = [np.std(x) if x else np.nan for x in white_balls]
        df['max'] = [np.max(x) if x else np.nan for x in white_balls]
        df['min'] = [np.min(x) if x else np.nan for x in white_balls]
        logging.info("Statistical measures extracted successfully.")
        return df
    except Exception as e:
        logging.error(f"Error occurred during statistical measures extraction: {e}")
        return None
    
def extract_combination_characteristics(df):
    logging.info("Extracting combination characteristics...")
    try:
        # Ensure columns 'w1' to 'w5' exist in the dataframe and they are of numeric types (filling None with 0)
        if not set(['w1', 'w2', 'w3', 'w4', 'w5']).issubset(df.columns):
            raise ValueError("Missing columns 'w1' to 'w5' in the DataFrame")

        # Fill in None values with 0
        df[['w1', 'w2', 'w3', 'w4', 'w5']] = df[['w1', 'w2', 'w3', 'w4', 'w5']].fillna(0)

        # Check if all values are numeric using the updated method
        if not df[['w1', 'w2', 'w3', 'w4', 'w5']].apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all():
            raise ValueError("Non-numeric values in columns 'w1' to 'w5' after filling with 0")
        
        white_balls = df[['w1', 'w2', 'w3', 'w4', 'w5']].values.tolist()
        df['even_count'] = [sum(1 for num in x if num % 2 == 0) for x in white_balls]
        df['odd_count'] = [sum(1 for num in x if num % 2 != 0) for x in white_balls]
        df['prime_count'] = [sum(1 for num in x if is_prime(num)) for x in white_balls]

        # Check if Mega Ball or Powerball exists
        if 'mega_ball' in df.columns:
            df['mb_even_count'] = df['mega_ball'].apply(lambda x: 1 if x % 2 == 0 else 0)
            df['mb_odd_count'] = df['mega_ball'].apply(lambda x: 1 if x % 2 != 0 else 0)
        if 'pb' in df.columns:
            df['pb_even_count'] = df['pb'].apply(lambda x: 1 if x % 2 == 0 else 0)
            df['pb_odd_count'] = df['pb'].apply(lambda x: 1 if x % 2 != 0 else 0)
        logging.info("Combination characteristics extracted successfully.")
        return df
    except Exception as e:
        logging.error(f"Error occurred during combination characteristics extraction: {e}")
        return None

def is_prime(num):
    """
    Checks if a number is prime.
    
    Args:
    - num: Number to check
    
    Returns:
    - bool: True if the number is prime, False otherwise
    """
    if num <= 1:
        return False
    for i in range(2, int(np.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

