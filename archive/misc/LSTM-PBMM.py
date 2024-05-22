import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the data into a pandas DataFrame
data_mm = pd.read_json('https://data.ny.gov/resource/5xaw-6ayf.json')
data_pb = pd.read_json('https://data.ny.gov/resource/d6yy-54nr.json')

# Preprocess the data if necessary (e.g., convert date column to datetime format)
data_mm['draw_date'] = pd.to_datetime(data_mm['draw_date'], format='%Y-%m-%dT00:00:00.000').dt.date
data_pb['draw_date'] = pd.to_datetime(data_pb['draw_date'], format='%Y-%m-%dT00:00:00.000').dt.date

# Convert 'winning_numbers' column from string to list of integers
data_mm['winning_numbers'] = data_mm['winning_numbers'].apply(lambda x: [int(num) for num in x.split()])
data_pb['winning_numbers'] = data_pb['winning_numbers'].apply(lambda x: [int(num) for num in x.split()])

# Merge the two datasets based on common columns
data = pd.concat([data_mm, data_pb])

# Preprocess the merged data further
data['year'] = data['draw_date'].dt.year
data['month'] = data['draw_date'].dt.month
data['day'] = data['draw_date'].dt.day
data['weekday'] = data['draw_date'].dt.weekday

# One-hot encode the winning numbers
unique_numbers = set(data['winning_numbers'].sum())
for num in unique_numbers:
    data['number_{}'.format(num)] = data['winning_numbers'].apply(lambda x: 1 if num in x else 0)

# Normalize the date and other numerical columns
scaler = MinMaxScaler()
data[['year', 'month', 'day', 'weekday', 'm', 'r']] = scaler.fit_transform(data[['year', 'month', 'day', 'weekday', 'm', 'r']])

# Prepare the input sequences and targets
input_sequences = []
targets = []
seq_length = 5  # You can adjust this sequence length based on the desired context window

for i in range(len(data) - seq_length):
    input_sequences.append(data[['year', 'month', 'day', 'weekday', 'm', 'r', 'number_0', 'number_1', 'number_2',
                                 'number_3', 'number_4', 'number_5', 'number_6', 'number_7', 'number_8', 'number_9']].values[i:i+seq_length])
    targets.append(data[['number_0', 'number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6',
                         'number_7', 'number_8', 'number_9']].values[i+seq_length])

input_sequences = np.array(input_sequences)
targets = np.array(targets)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_sequences, targets, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, 16), activation='relu'))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
