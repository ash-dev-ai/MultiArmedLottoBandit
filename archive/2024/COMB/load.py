# load.py
import pandas as pd
from sodapy import Socrata
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_org_mb_data(API_ENDPOINT, API_LIMIT):
    """
    Load data for org_mb into a DataFrame.
    """
    try:
        client = Socrata(API_ENDPOINT, None)
        results = client.get("5xaw-6ayf", limit=API_LIMIT)  
        df = pd.DataFrame.from_records(results)
        return df
    except Exception as e:
        logging.error(f"Failed to load data for org_mb: {e}")
        raise

def load_org_pb_data(API_ENDPOINT, API_LIMIT):
    """
    Load data for org_pb into a DataFrame.
    """
    try:
        client = Socrata(API_ENDPOINT, None)
        results = client.get("d6yy-54nr", limit=API_LIMIT)
        df = pd.DataFrame.from_records(results)
        return df
    except Exception as e:
        logging.error(f"Failed to load data for org_pb: {e}")
        raise
