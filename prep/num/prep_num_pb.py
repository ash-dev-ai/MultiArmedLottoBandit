# prep_num_pb.py
import pandas as pd
import logging
from prep.num.count import CountNumbers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PrepNumPB:
    def __init__(self, input_file='data/data_pb.csv', output_file='data/num_pb.csv'):
        self.input_file = input_file
        self.output_file = output_file

    def load_data(self):
        """Load the Powerball data from a CSV file."""
        self.data = pd.read_csv(self.input_file)
        logging.info(f"Loaded data from {self.input_file}")

    def prepare_counts(self):
        """Prepare the counts dataset."""
        self.load_data()
        counter = CountNumbers(self.data, self.output_file, num_range=69, numA_range=26)
        counter.prepare_counts()

if __name__ == "__main__":
    prepper = PrepNumPB()
    prepper.prepare_counts()