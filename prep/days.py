import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataDays:
    def __init__(self, data: pd.DataFrame, name: str):
        self.data = data
        self.name = name

    def add_day_column(self):
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
        self.data['day'] = self.data['weekday'].map(day_mapping)
        logging.info(f"Added day column to {self.name} data")
        return self.data

    def add_date_column(self):
        """Add a date column with integer values in the format YYYYMMDD."""
        self.data['date'] = pd.to_datetime(self.data['draw_date']).dt.strftime('%Y%m%d').astype(int)
        logging.info(f"Added date column to {self.name} data")
        return self.data

    def process(self):
        self.add_day_column()
        self.add_date_column()
        return self.data


def add_day_and_date_columns(data, name):
    days_processor = DataDays(data, name)
    return days_processor.process()
