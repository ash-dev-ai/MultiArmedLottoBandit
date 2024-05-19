# sum.py
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_sums(data, name):
    """Add numSum and totalSum columns to the data and log the operation."""
    data['numSum'] = data[['num1', 'num2', 'num3', 'num4', 'num5']].astype(int).sum(axis=1)
    data['totalSum'] = data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].astype(int).sum(axis=1)
    logging.info(f"Added numSum and totalSum columns to {name} data")

# No main function, this script will be called from prep_main.py


