# export.py
import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def export_data(pb_data, mb_data):
    """Export Powerball and Mega Millions data to CSV files and create a combined dataset."""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    pb_file = os.path.join(data_dir, 'data_pb.csv')
    mb_file = os.path.join(data_dir, 'data_mb.csv')
    combined_file = os.path.join(data_dir, 'data_combined.csv')
    
    # Export Powerball data
    pb_data.to_csv(pb_file, index=False)
    logging.info(f"Powerball data exported to {pb_file}")
    
    # Export Mega Millions data
    mb_data.to_csv(mb_file, index=False)
    logging.info(f"Mega Millions data exported to {mb_file}")
    
    # Create and export combined dataset
    combined_data = pd.concat([pb_data, mb_data], ignore_index=True)
    combined_data.to_csv(combined_file, index=False)
    logging.info(f"Combined data exported to {combined_file}")

# No main function, this script will be called from prep_main.py
