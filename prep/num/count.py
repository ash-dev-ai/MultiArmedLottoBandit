# count.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CountNumbers:
    def __init__(self, data, output_file, num_range, numA_range):
        self.data = data
        self.output_file = output_file
        self.num_range = num_range
        self.numA_range = numA_range

    def count_numbers(self):
        """Count occurrences of each number in num1-5 and numA columns."""
        count_df = pd.DataFrame(index=range(1, self.num_range + 1), columns=[f'count_num{i}' for i in range(1, 6)] + ['count_numA'])
        count_df = count_df.fillna(0)  # Initialize counts to 0

        for num in range(1, self.num_range + 1):
            count_df.at[num, 'count_num1'] = (self.data['num1'] == num).sum()
            count_df.at[num, 'count_num2'] = (self.data['num2'] == num).sum()
            count_df.at[num, 'count_num3'] = (self.data['num3'] == num).sum()
            count_df.at[num, 'count_num4'] = (self.data['num4'] == num).sum()
            count_df.at[num, 'count_num5'] = (self.data['num5'] == num).sum()
        for numA in range(1, self.numA_range + 1):
            count_df.at[numA, 'count_numA'] = (self.data['numA'] == numA).sum()

        self.counts_df = count_df
        logging.info("Counted occurrences of each number")

    def save_counts(self):
        """Save the counts to a CSV file."""
        self.counts_df.to_csv(self.output_file)
        logging.info(f"Saved counts to {self.output_file}")

    def prepare_counts(self):
        """Prepare the counts dataset."""
        self.count_numbers()
        self.save_counts()