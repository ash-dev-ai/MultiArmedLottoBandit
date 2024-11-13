import os
import pandas as pd
from trials.trial0 import Trial0
from trials.rewardPenaltySys import RewardPenaltySystem
from datetime import datetime

def save_predictions(predictions, rule_type, dataset_name):
    """
    Save the predictions to predictions.csv with the appropriate rule type and dataset name.
    Creates the file if it doesn't exist.
    """
    predictions_file = os.path.join('data', 'predictions.csv')
    
    predictions_df = pd.DataFrame(predictions)
    predictions_df['date'] = datetime.now().strftime('%Y-%m-%d')
    predictions_df['dataset'] = dataset_name
    predictions_df['type'] = rule_type

    # Enforce correct column order to avoid misalignment issues
    column_order = ['date', 'num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'next_draw_day', 'dataset', 'type']
    for col in column_order:
        if col not in predictions_df.columns:
            predictions_df[col] = None
    predictions_df = predictions_df[column_order]

    # Load existing predictions and append new ones
    if os.path.exists(predictions_file):
        existing_df = pd.read_csv(predictions_file)
        existing_df = existing_df[column_order]  # Ensure existing file also matches column order
        combined_df = pd.concat([existing_df, predictions_df], ignore_index=True)
    else:
        combined_df = predictions_df

    # Save to file with consistent column order
    combined_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to {predictions_file} for type {rule_type} and dataset {dataset_name}.")
    
def run_trial(trial_class, dataset_name, root_dir, trial_type):
    """
    Run the trial, generate predictions, adjust predictions, and save to CSV.
    """
    trial_instance = trial_class(dataset_name, root_dir, determine_num_range(dataset_name))
    data = trial_instance.load_data()
    meta_models = trial_instance.load_meta_models()

    if meta_models:
        model_predictions = trial_instance.generate_model_predictions(data, meta_models)
        last_draw_day = data['day'].values[-1]

        adjusted_predictions = trial_instance.generate_adjusted_predictions(model_predictions, last_draw_day)
        
        for prediction in adjusted_predictions:
            rule_type = prediction['type']
            save_predictions([prediction], rule_type, dataset_name)
        
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

def apply_reward_penalty_system(predictions_file, data_file):
    reward_system = RewardPenaltySystem(predictions_file, data_file)
    reward_system.update_scores()
    print(f"Scores updated in {predictions_file}.")

def main():
    datasets = ['combined', 'pb', 'mb']
    root_dir = 'D:/Projects/Project-PBMM/current/MultiArmedLottoBandit'

    for dataset in datasets:
        run_trial(Trial0, dataset, root_dir, trial_type='T-0')

        predictions_file = os.path.join(root_dir, 'data', 'predictions.csv')
        data_file = os.path.join(root_dir, 'data', f"data_{dataset}.csv")
        apply_reward_penalty_system(predictions_file, data_file)

if __name__ == "__main__":
    main()
