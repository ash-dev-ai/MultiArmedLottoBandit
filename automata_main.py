import os
import importlib
import pandas as pd

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

def save_predictions(predictions, dataset_name, rule_type):
    """
    Save the predictions to predictions.csv with the appropriate rule type.
    Creates the file if it doesn't exist.
    """
    predictions_file = os.path.join('data', 'predictions.csv')
    
    # Prepare predictions DataFrame
    predictions_df = pd.DataFrame(predictions)
    predictions_df['type'] = rule_type  # Add the rule type as A-<rule_number>

    # Save predictions to CSV, append if file exists
    if os.path.exists(predictions_file):
        predictions_df.to_csv(predictions_file, mode='a', header=False, index=False)
    else:
        predictions_df.to_csv(predictions_file, mode='w', header=True, index=False)

    print(f"Predictions saved to {predictions_file} for type {rule_type}.")

def generate_predictions_for_dataset(dataset_name, rule_class):
    data, dataset_name = load_dataset(dataset_name)

    if data is not None:
        # Determine the number range for this dataset
        num_range = determine_num_range(dataset_name)

        # Initialize the rule class with the number range
        rule_instance = rule_class(num_range=num_range)

        # Generate predictions using the rule
        predictions = rule_instance.generate_predictions(data, n_predictions=5)
        
        # Format predictions for saving
        formatted_predictions = []
        for prediction in predictions:
            formatted_predictions.append({
                'num1': prediction['num1-5'][0],
                'num2': prediction['num1-5'][1],
                'num3': prediction['num1-5'][2],
                'num4': prediction['num1-5'][3],
                'num5': prediction['num1-5'][4],
                'numA': prediction['numA']
            })

        # Determine the rule type code for saving (e.g., A-110, A-150)
        rule_type = f"A-{rule_class.__name__.replace('Rule', '')}"
        save_predictions(formatted_predictions, dataset_name, rule_type)

        # Output the predictions
        print(f"Generated Predictions for {dataset_name} using {rule_class.__name__}:")
        for i, prediction in enumerate(predictions, start=1):
            print(f"Prediction {i}: num1-5 = {prediction['num1-5']}, numA = {prediction['numA']}")
    else:
        print(f"No predictions for {dataset_name} due to missing or invalid data.")

def main():
    # List of dataset names
    datasets = ['data_combined.csv', 'data_pb.csv', 'data_mb.csv']
    
    # List of rule numbers
    rule_numbers = [30, 37, 42, 45, 73, 90, 110, 150, 254]
    
    # Dynamically import and apply each rule
    for rule_number in rule_numbers:
        rule_module_name = f"automata.Rule{rule_number}"
        try:
            rule_module = importlib.import_module(rule_module_name)
            rule_class = getattr(rule_module, f"Rule{rule_number}")
            
            for dataset in datasets:
                generate_predictions_for_dataset(dataset, rule_class)
                
        except (ModuleNotFoundError, AttributeError) as e:
            print(f"Error loading {rule_module_name}: {e}")

if __name__ == "__main__":
    main()
