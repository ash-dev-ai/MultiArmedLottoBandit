import os
import pandas as pd
from trials.trial0 import Trial0
from datetime import datetime

def save_predictions(predictions, rule_type, dataset_name):
    predictions_file = os.path.join('data', 'predictions.csv')
    predictions_df = pd.DataFrame(predictions)
    
    # Add rule_type and dataset_name directly to the respective columns
    predictions_df['type'] = rule_type
    predictions_df['dataset'] = dataset_name  # Add dataset column for clarity
    
    # Define consistent column order
    column_order = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'next_draw_day', 'type', 'dataset']
    predictions_df = predictions_df[column_order]
    
    if os.path.exists(predictions_file):
        predictions_df.to_csv(predictions_file, mode='a', header=False, index=False)
    else:
        predictions_df.to_csv(predictions_file, mode='w', header=True, index=False)
    
    print(f"Predictions saved to {predictions_file} for type {rule_type} and dataset {dataset_name}.")

def run_trial(trial_class, dataset_name, root_dir, trial_type):
    trial_instance = trial_class(dataset_name, root_dir, determine_num_range(dataset_name))
    data = trial_instance.load_data()
    meta_models = trial_instance.load_meta_models()
    
    if meta_models:
        model_predictions = trial_instance.generate_model_predictions(data, meta_models)
        last_draw_day = data['day'].values[-1]
        adjusted_predictions = trial_instance.generate_adjusted_predictions(model_predictions, last_draw_day)
        
        # Set the type as T-0 for trial predictions and T-0-A-RULE# for automata adjustments
        for rule_number in trial_instance.automata_rules:
            rule_type = f"{trial_type}-A-{rule_number}"
            save_predictions(adjusted_predictions, rule_type, dataset_name)
        
        print(f"Trial0 predictions saved for {dataset_name}")
    else:
        print("Meta-models could not be loaded. Exiting.")

def determine_num_range(dataset_name):
    if "pb" in dataset_name:
        return (69, 26)
    elif "mb" in dataset_name:
        return (70, 25)
    else:
        return (70, 26)

def main():
    datasets = ['combined', 'pb', 'mb']
    root_dir = 'D:/Projects/Project-PBMM/current/MultiArmedLottoBandit'
    
    for dataset in datasets:
        run_trial(Trial0, dataset, root_dir, trial_type='T-0')

if __name__ == "__main__":
    main()
