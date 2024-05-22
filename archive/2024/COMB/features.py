#features.py
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_date_features(df):
    """
    Extract date features from the date of the lottery draw.
    """
    try:
        # Example: Extract day of the week, month, and quarter
        df['day_of_week'] = df['draw_date'].dt.day_name()
        df['month'] = df['draw_date'].dt.month_name()
        df['quarter'] = df['draw_date'].dt.quarter
        # Add more date features as needed
        return df
    except Exception as e:
        logging.error(f"Failed to extract date features: {e}")
        raise

def extract_historical_patterns(df):
    """
    Analyze historical lottery data to identify patterns or trends.
    """
    try:
        # Example: Calculate frequency of each number being drawn over time
        number_columns = [col for col in df.columns if col.startswith('w')]
        for col in number_columns:
            df[f'{col}_frequency'] = df[col].map(df[col].value_counts())
        # Add more historical patterns features as needed
        return df
    except Exception as e:
        logging.error(f"Failed to extract historical patterns: {e}")
        raise

def extract_statistical_measures(df):
    """
    Calculate statistical measures from past lottery results.
    """
    try:
        # Example: Calculate mean, median, and standard deviation of drawn numbers
        number_columns = [col for col in df.columns if col.startswith('w')]
        df['mean_numbers'] = df[number_columns].mean(axis=1)
        df['median_numbers'] = df[number_columns].median(axis=1)
        df['std_deviation_numbers'] = df[number_columns].std(axis=1)
        # Add more statistical measures features as needed
        return df
    except Exception as e:
        logging.error(f"Failed to extract statistical measures: {e}")
        raise

def extract_number_relationships(df):
    """
    Explore relationships between different numbers drawn in the same lottery.
    """
    try:
        # Initialize a DataFrame to store the number relationships features
        number_relationships_features = pd.DataFrame()

        # Calculate correlation between pairs of numbers
        number_columns = [col for col in df.columns if col.startswith('w')]
        correlation_matrix = df[number_columns].corr()
        correlation_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        correlation_matrix = correlation_matrix.stack().reset_index()
        correlation_matrix.columns = ['number_1', 'number_2', 'correlation']
        number_relationships_features['correlation'] = correlation_matrix['correlation']

        # Previous occurrences of specific number sequences
        for i in range(1, 6):
            col_name = f'w{i}'
            df[col_name + '_sequence'] = df[col_name].astype(str).shift(1) + '-' + df[col_name].astype(str)
            number_relationships_features[col_name + '_sequence'] = df[col_name + '_sequence']
        
        # Differences between consecutive numbers
        for i in range(1, 5):
            col_name = f'w{i}'
            next_col_name = f'w{i+1}'
            df[col_name + '_' + next_col_name + '_difference'] = df[next_col_name] - df[col_name]
            number_relationships_features[col_name + '_' + next_col_name + '_difference'] = df[col_name + '_' + next_col_name + '_difference']

        # Ratios between consecutive numbers (avoiding division by zero)
        for i in range(1, 5):
            col_name = f'w{i}'
            next_col_name = f'w{i+1}'
            df[col_name + '_' + next_col_name + '_ratio'] = df[next_col_name] / (df[col_name] + 1)  # Adding 1 to avoid division by zero
            number_relationships_features[col_name + '_' + next_col_name + '_ratio'] = df[col_name + '_' + next_col_name + '_ratio']

        return number_relationships_features
    except Exception as e:
        logging.error(f"Failed to extract number relationships: {e}")
        raise

def extract_time_series_features(df):
    """
    Consider time series analysis techniques to capture temporal patterns.
    """
    try:
        time_series_features = {}

        # Perform time series decomposition
        decomposition = sm.tsa.seasonal_decompose(df['value'], model='additive')
        time_series_features['trend'] = decomposition.trend
        time_series_features['seasonality'] = decomposition.seasonal
        time_series_features['residual'] = decomposition.resid

        # Calculate autocorrelation and partial autocorrelation functions
        autocorrelation = sm.tsa.acf(df['value'], nlags=10, fft=True)
        partial_autocorrelation = sm.tsa.pacf(df['value'], nlags=10, method='ols')
        time_series_features['autocorrelation'] = autocorrelation
        time_series_features['partial_autocorrelation'] = partial_autocorrelation

        # Compute rolling averages or moving windows
        rolling_mean = df['value'].rolling(window=7).mean()
        rolling_std = df['value'].rolling(window=7).std()
        time_series_features['rolling_mean'] = rolling_mean
        time_series_features['rolling_std'] = rolling_std

        # Include lagged values
        df['lag_1'] = df['value'].shift(1)
        df['lag_2'] = df['value'].shift(2)
        time_series_features['lag_1'] = df['lag_1']
        time_series_features['lag_2'] = df['lag_2']

        return time_series_features
    except Exception as e:
        logging.error(f"Failed to extract time series features: {e}")
        raise

def extract_combination_characteristics(df):
    """
    Analyze characteristics of the lottery combinations themselves.
    """
    try:
        # Calculate sum of numbers in each combination
        df['sum_of_numbers'] = df[[col for col in df.columns if col.startswith('w')]].sum(axis=1)
        
        # Determine distribution of odd and even numbers
        odd_numbers = df[[col for col in df.columns if col.startswith('w')]].apply(lambda x: np.sum(x % 2 == 1), axis=1)
        even_numbers = df[[col for col in df.columns if col.startswith('w')]].apply(lambda x: np.sum(x % 2 == 0), axis=1)
        df['odd_numbers'] = odd_numbers
        df['even_numbers'] = even_numbers
        
        # Check for presence of consecutive or sequential numbers
        consecutive_count = df[[col for col in df.columns if col.startswith('w')]].apply(
            lambda row: sum(abs(row[i] - row[i+1]) == 1 for i in range(len(row)-1)), axis=1)
        df['consecutive_numbers'] = consecutive_count.apply(lambda x: 1 if x > 0 else 0)
        
        return df
    except Exception as e:
        logging.error(f"Failed to extract combination characteristics: {e}")
        raise


