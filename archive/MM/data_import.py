import warnings
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the data into a pandas DataFrame
data_mm = pd.read_json('https://data.ny.gov/resource/5xaw-6ayf.json')

# Fetch data from the API and convert to DataFrame
data_pb = pd.read_json('https://data.ny.gov/resource/d6yy-54nr.json')

#data_mm
# Preprocess the data if necessary (e.g., convert date column to datetime format)
data_mm['draw_date'] = pd.to_datetime(data_mm['draw_date']).dt.date
# Convert 'winning_numbers' column from string to list of integers
data_mm['winning_numbers'] = data_mm['winning_numbers'].apply(lambda x: [int(num) for num in x.split()])
# Rename Columns inplace
data_mm.rename(columns={'draw_date': 'd','mega_ball': 'r', 'multiplier': 'm', 'winning_numbers': 'w'}, inplace=True)

#data_pb
# Preprocess the data if necessary (e.g., convert date column to datetime format)
data_pb['draw_date'] = pd.to_datetime(data_pb['draw_date']).dt.date
# Convert 'winning_numbers' column from string to list of integers
data_pb['winning_numbers'] = data_pb['winning_numbers'].apply(lambda x: [int(num) for num in x.split()])
data_pb['w'] = data_pb['winning_numbers'].apply(lambda x: x[:5])
data_pb['r'] = data_pb['winning_numbers'].apply(lambda x: x[-1])
data_pb.rename(columns={'draw_date': 'd', 'multiplier': 'm'}, inplace=True)
data_pb.drop(columns=['winning_numbers'], inplace=True)

