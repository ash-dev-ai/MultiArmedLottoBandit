# prep_pb.py
import os
import pandas as pd
from datetime import datetime
import logging
from init.config import API_ENDPOINT, API_LIMIT, API_ENDPOINT_PB
from sodapy import Socrata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to store the DataFrame
pb_data = None

class PrepPB:
    def __init__(self):
        self.last_run_file = os.path.join('prep', 'last_run_pb.txt')
        self.client = Socrata(API_ENDPOINT, None)

    def load_data(self):
        """Fetch data from the API and load it into a DataFrame."""
        global pb_data
        results = self.client.get(API_ENDPOINT_PB, limit=API_LIMIT)
        pb_data = pd.DataFrame.from_records(results)
        logging.info(f"Fetched {len(pb_data)} records from PB API")

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
        global pb_data
        # Ensure the winning numbers are strings
        pb_data['winning_numbers'] = pb_data['winning_numbers'].astype(str)
        # Split the winning numbers into six separate columns
        split_cols = pb_data['winning_numbers'].str.split(expand=True)
        pb_data['num1'] = split_cols[0]
        pb_data['num2'] = split_cols[1]
        pb_data['num3'] = split_cols[2]
        pb_data['num4'] = split_cols[3]
        pb_data['num5'] = split_cols[4]
        pb_data['numA'] = split_cols[5]
        logging.info("Split winning numbers into separate columns")

    def drop_multiplier(self):
        """Drop the Multiplier column from the data."""
        global pb_data
        if 'multiplier' in pb_data.columns:
            pb_data.drop(columns=['multiplier'], inplace=True)
            logging.info("Dropped Multiplier column")

    def label_weekday(self):
        """Label each draw date with the corresponding weekday."""
        global pb_data
        pb_data['draw_date'] = pd.to_datetime(pb_data['draw_date'])
        pb_data['weekday'] = pb_data['draw_date'].dt.day_name()
        pb_data['weekday'] = pb_data['weekday'].apply(
            lambda x: 'Monday' if x == 'Monday' else ('Wednesday' if x == 'Wednesday' else 'Saturday')
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
    preparer = PrepPB()
    preparer.prepare_data()