# days.py
import logging
import pandas as pd

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

# No main function, this script will be called from prep_main.py
