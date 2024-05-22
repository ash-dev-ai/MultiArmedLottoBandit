import pandas as pd
import numpy as np

# Function 1: Drop the 'm' column from datasets
def drop_m_column(dataset):
    if 'm' in dataset.columns:
        dataset.drop('m', axis=1, inplace=True)

# Function 2: Convert 'd' column to days since the earliest date
def datetime_to_days_since_earliest(date, earliest_date):
    days_since_earliest = (date - earliest_date).days + 1
    if days_since_earliest < 1:
        days_since_earliest = 1
    return days_since_earliest

# Function 3: Update 'w' column and split it into w1, w2, w3, w4, and w5
def update_w_column(dataset):
    # Calculate the maximum number of elements in any 'w' list
    max_length = dataset['w'].apply(lambda x: len(x) if isinstance(x, list) else 1).max()

    # Create new columns for each element in the 'w' list
    for i in range(1, max_length + 1):
        col_name = f'w{i}'
        dataset[col_name] = dataset['w'].apply(lambda x: x[i - 1] if isinstance(x, list) and len(x) >= i else np.nan)

    # Drop the original 'w' column
    dataset.drop(columns=['w'], inplace=True)

    return dataset

# Function 4: Process 'all_tickets' dataset in chunks
def process_all_tickets_in_chunks(dataset, chunk_size=100000):
    total_rows = len(dataset)
    num_chunks = (total_rows // chunk_size) + 1

    for chunk_num in range(num_chunks):
        start_idx = chunk_num * chunk_size
        end_idx = (chunk_num + 1) * chunk_size
        chunk = dataset.iloc[start_idx:end_idx].copy()  # Create a copy of the chunk

        earliest_date = chunk["d"].min()
        chunk["d"] = chunk["d"].apply(lambda x: datetime_to_days_since_earliest(x, earliest_date))

        chunk = update_w_column(chunk)  # Update chunk using the modified function

        if chunk_num == 0:
            updated_dataset = chunk
        else:
            updated_dataset = pd.concat([updated_dataset, chunk])

    return updated_dataset

# Function to print dataset information
def print_dataset_info(dataset_name, dataset):
    print(f"\nDataset: {dataset_name}")
    print("First few rows:")
    print(dataset.head())
    print(f"\nShape of the dataset: {dataset.shape}")
    print("\nColumn names:")
    print(dataset.columns)
    print("\nData types of each column:")
    print(dataset.dtypes)
    print("\nNumber of non-null values in each column:")
    print(dataset.count())
    print("\nBasic statistics:")
    print(dataset.describe())
    print("\nNumber of missing values in each column:")
    print(dataset.isnull().sum())
    print("\nPercentage of missing values in each column:")
    print((dataset.isnull().sum() / len(dataset)) * 100)


