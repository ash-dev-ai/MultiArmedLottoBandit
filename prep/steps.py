# steps.py
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataSteps:
    def __init__(self, data: pd.DataFrame, name: str):
        self.data = data
        self.name = name

    def add_difference_columns(self):
        """Add columns for differences between specified number columns."""
        columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA']
        differences = [
            ('num1', 'num2'), ('num1', 'num3'), ('num1', 'num4'), ('num1', 'num5'), ('num1', 'numA'),
            ('num2', 'num3'), ('num2', 'num4'), ('num2', 'num5'), ('num2', 'numA'),
            ('num3', 'num4'), ('num3', 'num5'), ('num3', 'numA'),
            ('num4', 'num5'), ('num4', 'numA'),
            ('num5', 'numA')
        ]

        for col1, col2 in differences:
            new_col_name = f"N{col1[-1]}-{col2[-1]}"
            self.data[new_col_name] = self.data[col1].astype(int) - self.data[col2].astype(int)
            logging.info(f"Added column {new_col_name} to {self.name} data")
        return self.data


def add_step_columns(data, name):
    steps_processor = DataSteps(data, name)
    return steps_processor.add_difference_columns()


if __name__ == "__main__":
    logging.error("This script should be called from prep_main.py")
