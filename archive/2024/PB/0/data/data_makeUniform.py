import pandas as pd
import numpy as np
from fractions import Fraction

# Import the winning_data dictionary from data_winRules.py
from data_winRules import winning_data

# Function 1: Drop the 'm' column from datasets

def drop_m_column(dataset):
    if 'm' in dataset.columns:
        dataset.drop('m', axis=1, inplace=True)

# Function 2: Convert 'd' column to days since the earliest date
def datetime_to_days_since_earliest(d_column, earliest_date):
    days_since_earliest = (pd.to_datetime(d_column) - earliest_date).dt.days + 1
    days_since_earliest = days_since_earliest.apply(lambda x: max(x, 1))  # Ensure it's at least 1
    return days_since_earliest

# Function 3: Function to update the 'w' column and add new columns

def update_w_column(dataset):
    if 'w' in dataset.columns:
        max_length = dataset['w'].apply(lambda x: len(x) if isinstance(x, list) else 1).max()
        for i in range(1, max_length + 1):
            col_name = f'w{i}'
            dataset[col_name] = dataset['w'].apply(lambda x: x[i - 1] if isinstance(x, list) and len(x) >= i else None)
    else:
        pass

    # Add the 'Type', 'Desc', 'Probability', and 'Prize' columns from winning_data to the dataset
    win_type = 1  # Specify the win type you want to use
    dataset['win_type'] = win_type
    dataset['Desc'] = "5w+r"
    dataset['Probability'] = 1 / 292201338
    dataset['Prize'] = 20000000
    return dataset

def print_dataset_info(dataset):
    print("First few rows:")
    print(dataset.head())
    print(f"\nShape of the dataset: {dataset.shape}")
    print("\nColumn names:")
    print(dataset.columns.tolist())  # Convert columns to a list
    print("\nData types of each column:")
    print(dataset.dtypes)
    print("\nNumber of non-null values in each column:")
    print(dataset.count())
    print("\nBasic statistics:")
    print(dataset.describe())
    print("\nNumber of missing values in each column:")
    print(dataset.isnull().sum())
    missing_percentage = (dataset.isnull().sum() / len(dataset)) * 100
    print("\nPercentage of missing values in each column:")
    print(missing_percentage)


