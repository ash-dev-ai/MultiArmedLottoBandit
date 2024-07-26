import sys
import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import ParameterGrid
import time

# Get the absolute path of the Data folder
data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Data"))
# Get the absolute path of the Parameters folder
param_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Parameters"))
# Add the Parameters folder to sys.path
sys.path.append(param_folder)
# Add the Data folder to sys.path
sys.path.append(data_folder)

# Import necessary modules from the Data folder
from splitPairs import dataset_pairs
from Datasets import org_mm, org_pb, full_mm, full_pb, new_mm, new_pb
# Import the param_grid_rf_combinations from paramGrid.py
from paramGrid import param_grid_mnnb_combinations

# Record the start time
start_time = time.process_time()

def datetime_to_days_since_earliest(date_str, earliest_date):
    date = pd.to_datetime(date_str)
    return (date - earliest_date).days

def MnNB(X, y, col_name):
    # Create a list to store the trained models for each target column
    models = []
    for params in param_grid_mnnb_combinations:
        # Create a MultinomialNB with current hyperparameters
        model = MultinomialNB()

        # Print the column name and hyperparameters for clarity during training
        print(f"Training model for column: {col_name} with Hyperparameters: {params}")

        # Fit the model to the data using partial_fit (incremental learning)
        # The model is first initialized with classes=[0, 1]
        if len(np.unique(y)) == 2:
            model.partial_fit(X, y, classes=[0, 1])  # Assuming y contains binary labels 0 and 1
        else:
            # If there are more than two classes, use unique classes from y as classes argument
            model.partial_fit(X, y, classes=np.unique(y))

        # Print the training completion message
        print(f"Model training for column '{col_name}' with Hyperparameters: {params} is complete.")

        # Add the trained model to the list
        models.append(model)

    # Return the list of trained models for all hyperparameter combinations
    return models

# Create a list of all six datasets
datasets = [org_mm, org_pb, full_mm, full_pb, new_mm, new_pb]

def run_mnnb():
    all_mnnb = []
    for dataset in datasets:
        dataset_mnnb = []
        for col in dataset.columns:
            dataset_mnnb.extend(MnNB(dataset.drop(columns=[col]), dataset[col], col))
        all_mnnb.append(dataset_mnnb)
    return all_mnnb

# Modify the variable name to store all mnnb models in a nested list
all_mnnb = run_mnnb()

# Unpack the models for each dataset
mnnb_mm, mnnb_pb, mnnb_full_mm, mnnb_full_pb, mnnb_new_mm, mnnb_new_pb = all_mnnb

# Calculate and print the elapsed time
end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time:.4f} seconds")
