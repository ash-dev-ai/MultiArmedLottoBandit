# days.py
import logging
import pandas as pd
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_day_column(data, name):
    """Add a day column with numeric values representing the day of the week."""
    day_mapping = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7
    }
    data['day'] = data['weekday'].map(day_mapping)
    logging.info(f"Added day column to {name} data")

def add_date_column(data, name):
    """Add a date column with integer values in the format YYYYMMDD."""
    data['date'] = pd.to_datetime(data['draw_date']).dt.strftime('%Y%m%d').astype(int)
    logging.info(f"Added date column to {name} data")

# Determine the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct paths relative to the script directory
data_combined_path = os.path.join(script_dir, '..', 'data', 'data_combined.csv')
data_pb_path = os.path.join(script_dir, '..', 'data', 'data_pb.csv')
data_mb_path = os.path.join(script_dir, '..', 'data', 'data_mb.csv')

# Load the datasets from the constructed paths
data_combined = pd.read_csv(data_combined_path)
data_pb = pd.read_csv(data_pb_path)
data_mb = pd.read_csv(data_mb_path)

# Create a dictionary to process each dataset
datasets = {
    'combined': data_combined,
    'pb': data_pb,
    'mb': data_mb
}

# Apply the transformations to each dataset
for name, dataset in datasets.items():
    add_day_column(dataset, name)
    add_date_column(dataset, name)

# Example: Display the modified 'combined' dataset to verify changes
print(datasets['combined'].head())
