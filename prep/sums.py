# sum.py
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataSums:
    def __init__(self, data: pd.DataFrame, name: str):
        self.data = data
        self.name = name

    def add_sums(self):
        """Add numSum and totalSum columns to the data."""
        self.data['numSum'] = self.data[['num1', 'num2', 'num3', 'num4', 'num5']].astype(int).sum(axis=1)
        self.data['totalSum'] = self.data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].astype(int).sum(axis=1)
        logging.info(f"Added numSum and totalSum columns to {self.name} data")
        return self.data


def add_sums(data, name):
    sums_processor = DataSums(data, name)
    return sums_processor.add_sums()


if __name__ == "__main__":
    logging.error("This script should be called from prep_main.py")
