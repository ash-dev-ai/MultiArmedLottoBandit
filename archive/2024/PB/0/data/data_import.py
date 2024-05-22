import warnings
import pandas as pd
from sodapy import Socrata


# Suppress warnings
warnings.filterwarnings("ignore")

# Fetch data from the API and convert to DataFrame
client = Socrata("data.ny.gov", None)

results = client.get("d6yy-54nr", limit=5000)

data_pb = pd.DataFrame.from_records(results)

# Preprocess the data if necessary (e.g., convert date column to datetime format)
data_pb['draw_date'] = pd.to_datetime(data_pb['draw_date']).dt.date

# Convert 'winning_numbers' column from string to list of integers
data_pb['winning_numbers'] = data_pb['winning_numbers'].apply(lambda x: [int(num) for num in x.split()])
data_pb['w'] = data_pb['winning_numbers'].apply(lambda x: x[:5])
data_pb['r'] = data_pb['winning_numbers'].apply(lambda x: x[-1])
data_pb.rename(columns={'draw_date': 'd', 'multiplier': 'm'}, inplace=True)
data_pb.drop(columns=['winning_numbers'], inplace=True)

