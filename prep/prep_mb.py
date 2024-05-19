# prep_mb.py
import os
import pandas as pd
from datetime import datetime
import logging
from init.config import API_ENDPOINT, API_LIMIT, API_ENDPOINT_MM
from sodapy import Socrata

# Ensure the log directory exists
log_dir = os.path.join('prep', 'log')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PrepMB:
    def __init__(self):
        self.last_run_file = os.path.join(log_dir, 'last_run_mb.txt')
        self.client = Socrata(API_ENDPOINT, None)
        self.data = None

    def load_data(self):
        """Fetch data from the API and load it into a DataFrame."""
        results = self.client.get(API_ENDPOINT_MM, limit=API_LIMIT)
        self.data = pd.DataFrame.from_records(results)
        logging.info(f"Fetched {len(self.data)} records from MM API")

    def record_last_run(self):
        """Record the current timestamp as the last run time."""
        with open(self.last_run_file, 'w') as f:
            f.write(datetime.now().isoformat())
        logging.info(f"Recorded last run at {datetime.now().isoformat()}")

    def read_last_run(self):
        """Read the last run timestamp from the file."""
        if os.path.exists(self.last_run_file):
            with open(self.last_run_file, 'r') as f:
                last_run = f.read().strip()
            logging.info(f"Last run was at {last_run}")
            return last_run
        else:
            logging.info("No previous run recorded.")
            return None

    def split_winning_numbers(self):
        """Split the winning numbers into separate columns."""
        # Ensure the winning numbers are strings
        self.data['winning_numbers'] = self.data['winning_numbers'].astype(str)
        # Split the winning numbers into five separate columns
        split_cols = self.data['winning_numbers'].str.split(expand=True)
        self.data['num1'] = split_cols[0]
        self.data['num2'] = split_cols[1]
        self.data['num3'] = split_cols[2]
        self.data['num4'] = split_cols[3]
        self.data['num5'] = split_cols[4]
        self.data['numA'] = self.data['mega_ball']
        self.data.drop(columns=['mega_ball'], inplace=True)
        logging.info("Split winning numbers into separate columns and renamed mega_ball to numA")

    def drop_multiplier(self):
        """Drop the Multiplier column from the data."""
        if 'multiplier' in self.data.columns:
            self.data.drop(columns=['multiplier'], inplace=True)
            logging.info("Dropped Multiplier column")

    def label_weekday(self):
        """Label each draw date with the corresponding weekday."""
        self.data['draw_date'] = pd.to_datetime(self.data['draw_date'])
        self.data['weekday'] = self.data['draw_date'].dt.day_name()
        self.data['weekday'] = self.data['weekday'].apply(
            lambda x: 'Tuesday' if x == 'Tuesday' else ('Friday' if x == 'Friday' else 'Invalid')
        )
        logging.info("Labeled draw dates with the corresponding weekday")

    def prepare_data(self):
        """Execute the data preparation steps."""
        self.load_data()
        self.split_winning_numbers()
        self.drop_multiplier()
        self.label_weekday()
        self.record_last_run()

    def get_data(self):
        """Return the prepared data."""
        return self.data
    
#Main here if need to see variable in variable explorer
if __name__ == "__main__":
    preparer = PrepMB()
    preparer.prepare_data()
    print(preparer.get_data().head())