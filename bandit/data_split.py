# data_split.py
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, test_size=0.2, val_size=0.1):
    """Split the data into training, validation, and test sets."""
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size / (1 - test_size), random_state=42)
    return train_data, val_data, test_data

if __name__ == "__main__":
    # Load the datasets
    data_combined = pd.read_csv('../data/data_combined.csv')
    data_pb = pd.read_csv('../data/data_pb.csv')
    data_mb = pd.read_csv('../data/data_mb.csv')
    
    # Split the datasets
    train_combined, val_combined, test_combined = split_data(data_combined)
    train_pb, val_pb, test_pb = split_data(data_pb)
    train_mb, val_mb, test_mb = split_data(data_mb)
    
    # Save the split datasets
    train_combined.to_csv('../data/train_combined.csv', index=False)
    val_combined.to_csv('../data/val_combined.csv', index=False)
    test_combined.to_csv('../data/test_combined.csv', index=False)
    
    train_pb.to_csv('../data/train_pb.csv', index=False)
    val_pb.to_csv('../data/val_pb.csv', index=False)
    test_pb.to_csv('../data/test_pb.csv', index=False)
    
    train_mb.to_csv('../data/train_mb.csv', index=False)
    val_mb.to_csv('../data/val_mb.csv', index=False)
    test_mb.to_csv('../data/test_mb.csv', index=False)
