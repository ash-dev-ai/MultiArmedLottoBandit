import pandas as pd
import numpy as np
from data_createMissing import pb, new_entries_pb
from data_import import data_pb
from data_makeUniform import drop_m_column, update_w_column, datetime_to_days_since_earliest, print_dataset_info

# API datasets
org_pb = data_pb.copy()

# Create new variables full_pb
full_pb = pb.copy()

# Return the modified datasets as new variables
new_pb = new_entries_pb.copy()

# Create a list of datasets
datasets = [org_pb, full_pb, new_pb]

# Loop through each dataset
for dataset in datasets:
    # Call the functions on each dataset
    drop_m_column(dataset)
    update_w_column(dataset)
    
    # Assuming you have 'earliest_date' defined as a datetime
    earliest_date = pd.to_datetime('2010-02-03')
    
    # Assuming 'd' is the date column in your dataset
    dataset['d'] = datetime_to_days_since_earliest(dataset['d'], earliest_date)
    
    # Print the dataset
    print_dataset_info(dataset)