# feature_engineering.py
import pandas as pd

def prepare_features(data):
    """Prepare features for the bandit algorithms."""
    features = data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day']]
    return features

if __name__ == "__main__":
    # Load the datasets
    data_combined = pd.read_csv('data/data_combined.csv')
    data_pb = pd.read_csv('data/data_pb.csv')
    data_mb = pd.read_csv('data/data_mb.csv')
    
    # Prepare features
    features_combined = prepare_features(data_combined)
    features_pb = prepare_features(data_pb)
    features_mb = prepare_features(data_mb)
    
    # Save the prepared features
    features_combined.to_csv('data/features_combined.csv', index=False)
    features_pb.to_csv('data/features_pb.csv', index=False)
    features_mb.to_csv('data/features_mb.csv', index=False)

