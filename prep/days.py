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

# Check if the files exist before attempting to read them
datasets = {}
if os.path.exists(data_combined_path):
    data_combined = pd.read_csv(data_combined_path)
    datasets['combined'] = data_combined
else:
    logging.warning(f"File not found: {data_combined_path}")

if os.path.exists(data_pb_path):
    data_pb = pd.read_csv(data_pb_path)
    datasets['pb'] = data_pb
else:
    logging.warning(f"File not found: {data_pb_path}")

if os.path.exists(data_mb_path):
    data_mb = pd.read_csv(data_mb_path)
    datasets['mb'] = data_mb
else:
    logging.warning(f"File not found: {data_mb_path}")

# Apply the transformations to each dataset if loaded successfully
for name, dataset in datasets.items():
    add_day_column(dataset, name)
    add_date_column(dataset, name)

# Example: Display the modified datasets to verify changes
for name, dataset in datasets.items():
    print(f"{name} dataset head:\n{dataset.head()}")
