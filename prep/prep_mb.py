# prep_mb.py
import os
import pandas as pd
from datetime import datetime
import logging
from init.config import API_ENDPOINT, API_LIMIT, API_ENDPOINT_MM
from sodapy import Socrata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to store the DataFrame
mb_data = None

class PrepMB:
    def __init__(self):
        self.last_run_file = os.path.join('prep', 'last_run_mb.txt')
        self.client = Socrata(API_ENDPOINT, None)

    def load_data(self):
        """Fetch data from the API and load it into a DataFrame."""
        global mb_data
        results = self.client.get(API_ENDPOINT_MM, limit=API_LIMIT)
        mb_data = pd.DataFrame.from_records(results)
        logging.info(f"Fetched {len(mb_data)} records from MM API")

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
        global mb_data
        # Ensure the winning numbers are strings
        mb_data['winning_numbers'] = mb_data['winning_numbers'].astype(str)
        # Split the winning numbers into five separate columns
        split_cols = mb_data['winning_numbers'].str.split(expand=True)
        mb_data['num1'] = split_cols[0]
        mb_data['num2'] = split_cols[1]
        mb_data['num3'] = split_cols[2]
        mb_data['num4'] = split_cols[3]
        mb_data['num5'] = split_cols[4]
        mb_data['numA'] = mb_data['mega_ball']
        mb_data.drop(columns=['mega_ball'], inplace=True)
        logging.info("Split winning numbers into separate columns and renamed mega_ball to numA")

    def drop_multiplier(self):
        """Drop the Multiplier column from the data."""
        global mb_data
        if 'multiplier' in mb_data.columns:
            mb_data.drop(columns=['multiplier'], inplace=True)
            logging.info("Dropped Multiplier column")

    def label_weekday(self):
        """Label each draw date with the corresponding weekday."""
        global mb_data
        mb_data['draw_date'] = pd.to_datetime(mb_data['draw_date'])
        mb_data['weekday'] = mb_data['draw_date'].dt.day_name()
        mb_data['weekday'] = mb_data['weekday'].apply(
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

if __name__ == "__main__":
    preparer = PrepMB()
    preparer.prepare_data()