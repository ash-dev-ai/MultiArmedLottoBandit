# automata_main.py
import os
import pandas as pd
from automata.Rule110 import Rule110
from automata.Rule150 import Rule150

def load_dataset(dataset_name):
    dataset_dir = 'data'
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path), dataset_name
    else:
        print(f"Warning: {dataset_path} not found.")
        return None, None

def determine_num_range(dataset_name):
    """
    Returns the appropriate number ranges based on the dataset.
    """
    if "pb" in dataset_name:  # Powerball
        return (69, 26)
    elif "mb" in dataset_name:  # Mega Millions
        return (70, 25)
    else:  # Combined or general
        return (70, 26)

def generate_predictions_for_dataset(dataset_name, rule_class):
    data, dataset_name = load_dataset(dataset_name)

    if data is not None:
        # Determine the number range for this dataset
        num_range = determine_num_range(dataset_name)

        # Initialize the rule class with the number range
        rule_instance = rule_class(num_range=num_range)

        # Generate 3 predictions using the rule
        predictions = rule_instance.generate_predictions(data, n_predictions=3)

        # Output the predictions
        print(f"Generated Predictions for {dataset_name} using {rule_class.__name__}:")
        for i, prediction in enumerate(predictions, start=1):
            print(f"Prediction {i}: num1-5 = {prediction['num1-5']}, numA = {prediction['numA']}")
    else:
        print(f"No predictions for {dataset_name} due to missing or invalid data.")

def main():
    # List of dataset names
    datasets = ['data_combined.csv', 'data_pb.csv', 'data_mb.csv']

    # Generate predictions for each dataset using Rule 110
    for dataset in datasets:
        generate_predictions_for_dataset(dataset, Rule110)
        generate_predictions_for_dataset(dataset, Rule150)

if __name__ == "__main__":
    main()
