import pandas as pd
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error

from MachinePairs import mm, pb

from Transformer import Transformer
from CNN import CNNModel

class PatternMachine:
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels

    def get_features(self):
        return self.features

    def get_labels(self):
        return self.labels
    
    def train_and_evaluate_transformer(self, X, y):
      
        # One-hot encode the target labels
        encoder = OneHotEncoder(sparse=False)
        y_onehot = encoder.fit_transform(y.values.reshape(-1, 1))

        transformer_model = Transformer(input_dim=X.shape[1], output_dim=128, max_sequence_length=X.shape[1], num_classes=y_onehot.shape[1])
        test_accuracy, mse = transformer_model.try_all_transformer_scenarios(X, y_onehot)
        return test_accuracy, mse

    def train_and_evaluate_cnn(self, X, y):
        # One-hot encode the target labels
        encoder = LabelEncoder()
        y_onehot = tf.keras.utils.to_categorical(encoder.fit_transform(y), num_classes=len(np.unique(y)))

        input_dim = X.shape[1]
        num_classes = y_onehot.shape[1]

        cnn_model = CNNModel(output_dim=128, max_sequence_length=input_dim, num_classes=num_classes)

        # Pass scenarios dictionary to the try_all_cnn_scenarios method
        test_accuracy, mse = cnn_model.try_all_cnn_scenarios(X, y_onehot, scenarios)  # <-- Add 'scenarios' as an argument
        return test_accuracy, mse

# Load your data here
# Create copies of the original DataFrames
com_mm = mm.copy()
com_pb = pb.copy()

# Function to get the nth element from the list, return None if the index is out of range
def get_nth_element(lst, n):
    if isinstance(lst, list) and len(lst) > n:
        return lst[n]
    else:
        return None

# Split the "w" column into individual columns w1, w2, w3, w4, w5
for i in range(5):
    com_mm["w{}".format(i+1)] = com_mm["w"].apply(lambda x: get_nth_element(x, i))
    com_pb["w{}".format(i+1)] = com_pb["w"].apply(lambda x: get_nth_element(x, i))

# Define your input data X and target variable y for different scenarios
#X_scenario1 = com_mm[['w1', 'w2', 'w3', 'w4', 'w5']]
#y_scenario1 = com_mm['d']

X_scenario2 = com_mm[['w1', 'w2', 'w3', 'w4', 'w5']]
y_scenario2 = com_mm['r']

X_scenario3 = com_mm[['w1', 'w2', 'w3', 'w4', 'w5', 'r']]
y_scenario3 = com_mm['d']

X_scenario4 = com_mm[['r']]
y_scenario4 = com_mm['d']

# Define the scenarios dictionary here
scenarios = {
    #"Scenario 1": {"features": X_scenario1, "labels": y_scenario1},
    "Scenario 2": {"features": X_scenario2, "labels": y_scenario2},
    "Scenario 3": {"features": X_scenario3, "labels": y_scenario3},
    "Scenario 4": {"features": X_scenario4, "labels": y_scenario4},
}

# Create a list to store the results for each scenario and model
results = {}

# Record the start time
start_time = time.process_time()

# Perform training and evaluation for each scenario using CNN and Transformer
for scenario_name, scenario_data in scenarios.items():
    print(f"\nTraining and evaluating {scenario_name} using CNN...")
    machine = PatternMachine(scenario_data["features"], labels=scenario_data["labels"])
    X = machine.get_features()
    y = machine.get_labels()

    cnn_test_accuracy, cnn_mse = machine.train_and_evaluate_cnn(X, y)
    transformer_test_accuracy, transformer_mse = machine.train_and_evaluate_transformer(X, y)

    results[scenario_name] = {
        "Transformer Test Accuracy": transformer_test_accuracy,
        "Transformer Mean Squared Error": transformer_mse,
        "CNN Test Accuracy": cnn_test_accuracy,
        "CNN Mean Squared Error": cnn_mse

    }

    print(f"CNN Test Accuracy for {scenario_name}: {cnn_test_accuracy:.4f}")
    print(f"CNN Mean Squared Error for {scenario_name}: {cnn_mse:.4f}")
    print(f"Transformer Test Accuracy for {scenario_name}: {transformer_test_accuracy:.4f}")
    print(f"Transformer Mean Squared Error for {scenario_name}: {transformer_mse:.4f}")

# Calculate and display the elapsed time
elapsed_time = time.process_time() - start_time
print("\nElapsed Time: {:.4f} seconds".format(elapsed_time))

# Print the evaluation results for each scenario and model
for scenario_name, scenario_results in results.items():
    print(f"\nResults for {scenario_name}:")
    print("CNN Test Accuracy:", scenario_results["CNN Test Accuracy"])
    print("CNN Mean Squared Error:", scenario_results["CNN Mean Squared Error"])
    print("Transformer Test Accuracy:", scenario_results["Transformer Test Accuracy"])
    print("Transformer Mean Squared Error:", scenario_results["Transformer Mean Squared Error"])

