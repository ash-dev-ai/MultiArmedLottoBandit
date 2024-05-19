# feature_engineering.py
import logging

def feature_engineering(data, name):
    """Create new features in the dataset."""
    data['range'] = data[['num1', 'num2', 'num3', 'num4', 'num5']].astype(int).max(axis=1) - data[['num1', 'num2', 'num3', 'num4', 'num5']].astype(int).min(axis=1)
    logging.info(f"Feature Engineering: Added 'range' feature to {name} dataset")

# No main function, this script will be called from explore_main.py



