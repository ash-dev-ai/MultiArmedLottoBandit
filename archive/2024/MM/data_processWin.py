import pandas as pd
import numpy as np
from data_createMissing import mm, pb, new_entries_pb, new_entries_mm
from data_import import data_mm, data_pb
from data_makeUniform import drop_m_column, update_w_column, print_dataset_info

# API datasets
org_mm = data_mm.copy()
org_pb = data_pb.copy()

# Create new variables for full_mm and full_pb
full_mm = mm.copy()
full_pb = pb.copy()

# Return the modified datasets as new variables
new_mm = new_entries_mm.copy()
new_pb = new_entries_pb.copy()

# Drop 'm' column from all datasets
datasets = [org_mm, org_pb, full_mm, full_pb, new_mm, new_pb]
for dataset in datasets:
    drop_m_column(dataset)

# Update 'w' column and split it for all datasets
all_datasets = [org_mm, org_pb, full_mm, full_pb, new_mm, new_pb]
for dataset in all_datasets:
    update_w_column(dataset)

# Print information for each dataset
print_dataset_info("org_mm", org_mm)
print_dataset_info("org_pb", org_pb)
print_dataset_info("full_mm", full_mm)
print_dataset_info("full_pb", full_pb)
print_dataset_info("new_mm", new_mm)
print_dataset_info("new_pb", new_pb)