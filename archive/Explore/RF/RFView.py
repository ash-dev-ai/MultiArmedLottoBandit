import sys
import os
# Get the absolute path of the Data folder
data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Data"))
# Get the absolute path of the Parameters folder
param_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Parameters"))
# Add the Parameters folder to sys.path
sys.path.append(param_folder)
# Add the Data folder to sys.path
sys.path.append(data_folder)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from splitPairs import dataset_pairs
from RF import all_mnnb, datasets

# Create a dictionary that maps dataset names to datasets
datasets_dict = {
    'org_mm_pairs': datasets[0],
    'org_pb_pairs': datasets[1],
    'full_mm_pairs': datasets[2],
    'full_pb_pairs': datasets[3],
    'new_mm_pairs': datasets[4],
    'new_pb_pairs': datasets[5],
}

# Create a dictionary that maps column names to models
column_models_dict = {}
for dataset_name, (mm_pairs, pb_pairs) in dataset_pairs.items():
    for col_name, _ in mm_pairs + pb_pairs:
        column_models_dict[col_name] = [model for model in mm_pairs + pb_pairs if model.col_name == col_name][0]

def plot_predicted_probabilities(dataset_name, column_name, model, dataset):
     # Extract the feature names used during the fit
    feature_names_fit = dataset.drop(columns=[column_name]).columns.tolist()

    # Get the feature matrix for visualization
    X_visualize = dataset.drop(columns=[column_name])

    # Predict the probabilities for the target column
    predicted_probabilities = model.predict_proba(X_visualize)

    # Plot the predicted probabilities
    plt.figure(figsize=(10, 6))
    sns.histplot(predicted_probabilities[:, 1], kde=True, color='b')
    plt.xlabel('Probability of Class 1')
    plt.ylabel('Frequency')
    plt.title(f'Predicted Probabilities for Column "{column_name}" in Dataset "{dataset_name}"')
    plt.show()

# Loop through each dataset and its corresponding target columns
for dataset_name, target_columns in dataset_pairs.items():
    # Get the corresponding dataset
    dataset = datasets_dict[dataset_name]

    # Loop through each target column
    for target_column in target_columns:
        # Extract the column name
        column_name = target_column[0]

        # Get the corresponding model for the current column
        model = column_models_dict[column_name]

        # Plot the predicted probabilities for the current dataset, column, and model
        plot_predicted_probabilities(dataset_name, column_name, model, dataset)