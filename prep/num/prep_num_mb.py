# prep_num_mb.py
import pandas as pd
import logging
from prep.num.count import CountNumbers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PrepNumMB:
    def __init__(self, input_file='data/data_mb.csv', output_file='data/num_mb.csv'):
        self.input_file = input_file
        self.output_file = output_file

    def load_data(self):
        """Load the Mega Millions data from a CSV file."""
        self.data = pd.read_csv(self.input_file)
        logging.info(f"Loaded data from {self.input_file}")

    def prepare_counts(self):
        """Prepare the counts dataset."""
        self.load_data()
        counter = CountNumbers(self.data, self.output_file, num_range=70, numA_range=25)
        counter.prepare_counts()

if __name__ == "__main__":
    prepper = PrepNumMB()
    prepper.prepare_counts()