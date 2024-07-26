import pandas as pd
import time

# Record the start time
start_time = time.process_time()

# 1) Import datasets from Datasets.py
from data_makeUniform import org_mm, org_pb, full_mm, full_pb, new_mm, new_pb

# 2) Split datasets into column pairs against 'd'
def split_dataset_into_pairs(df):
    pairs = []
    for column in df.columns:
        if column != 'd':
            pairs.append((column, 'd'))
    return pairs

def get_pairs_info(df, dataset_name):
    pairs = split_dataset_into_pairs(df)
    print(f"Pairs for {dataset_name}:")
    for pair in pairs:
        print(f"({pair[0]}, {pair[1]})")
    return pairs

# Example datasets:
datasets = {
    'org_mm': org_mm,
    'org_pb': org_pb,
    'full_mm': full_mm,
    'full_pb': full_pb,
    'new_mm': new_mm,
    'new_pb': new_pb,
}

# Dictionary to store pairs for each dataset
dataset_pairs = {}

# 3) Print function for basic information about the pairs
for dataset_name, dataset_df in datasets.items():
    pairs = get_pairs_info(dataset_df, dataset_name)
    dataset_pairs[f"{dataset_name}_pairs"] = pairs

# Calculate and print the elapsed time
end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time:.4f} seconds")

# Printing the returned pairs for each dataset
print("\nReturned pairs:")
for dataset_name, pairs in dataset_pairs.items():
    print(f"{dataset_name}: {pairs}")