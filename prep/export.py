# export.py
import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataExporter:
    def __init__(self, pb_data: pd.DataFrame, mb_data: pd.DataFrame, data_dir: str = 'data'):
        self.pb_data = pb_data
        self.mb_data = mb_data
        self.data_dir = data_dir
        self.pb_file = os.path.join(self.data_dir, 'data_pb.csv')
        self.mb_file = os.path.join(self.data_dir, 'data_mb.csv')
        self.combined_file = os.path.join(self.data_dir, 'data_combined.csv')

    def export_to_csv(self):
        """Export Powerball and Mega Millions data to CSV files and create a combined dataset."""
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Export Powerball data
        self.pb_data.to_csv(self.pb_file, index=False)
        logging.info(f"Powerball data exported to {self.pb_file}")
        
        # Export Mega Millions data
        self.mb_data.to_csv(self.mb_file, index=False)
        logging.info(f"Mega Millions data exported to {self.mb_file}")
        
        # Create and export combined dataset
        combined_data = pd.concat([self.pb_data, self.mb_data], ignore_index=True)
        combined_data.to_csv(self.combined_file, index=False)
        logging.info(f"Combined data exported to {self.combined_file}")
        return combined_data


def export_data(pb_data, mb_data):
    exporter = DataExporter(pb_data, mb_data)
    return exporter.export_to_csv()
