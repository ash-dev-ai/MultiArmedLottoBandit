# feature_engineering.py
import logging
import os
import pandas as pd

class FeatureEngineering:
    def __init__(self, data: pd.DataFrame, name: str):
        self.data = data
        self.name = name
    
    def add_range_feature(self):
        """Add range feature to the dataset."""
        self.data['range'] = self.data[['num1', 'num2', 'num3', 'num4', 'num5']].astype(int).max(axis=1) - self.data[['num1', 'num2', 'num3', 'num4', 'num5']].astype(int).min(axis=1)
        logging.info(f"Feature Engineering: Added 'range' feature to {self.name} dataset")
    
    def save_engineered_features(self, output_dir='explore/engineered_data'):
        """Save the dataset with engineered features."""
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'{output_dir}/{self.name}_engineered.csv'
        self.data.to_csv(output_file, index=False)
        logging.info(f"Engineered dataset saved at {output_file}")
    
    def engineer_features(self):
        """Run all feature engineering methods."""
        self.add_range_feature()
        self.save_engineered_features()
        return self.data

# No main function, this script will be called from explore_main.py
