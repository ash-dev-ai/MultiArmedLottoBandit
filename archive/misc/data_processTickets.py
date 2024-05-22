import os
import pandas as pd
from data_makeUniform import process_all_tickets_in_chunks

# List of data file paths
data_files = [
    'tickets/data_tickets_1.csv',
    'tickets/data_tickets_2.csv',
    'tickets/data_tickets_3.csv',
    'tickets/data_tickets_4.csv',
    'tickets/data_tickets_5.csv',
    'tickets/data_tickets_6.csv'
]

# Function to process data and create new CSV
def process_and_save(file_path):
    # Read the CSV
    dataset = pd.read_csv(file_path)
    
    # Convert 'd' column to datetime format
    dataset['d'] = pd.to_datetime(dataset['d'])
    
    # Calculate the earliest date
    earliest_date = dataset['d'].min()
    
    # Process the data
    dataset = process_all_tickets_in_chunks(dataset)
    
    # Extract file name and extension
    file_name, file_extension = os.path.splitext(file_path)
    
    # Create a new file path with "_proc" added to the filename
    new_file_path = f"{file_name}_proc{file_extension}"
    
    # Save the processed data to the new CSV file
    dataset.to_csv(new_file_path, index=False)

# Process and save each data file
for file_path in data_files:
    process_and_save(file_path)