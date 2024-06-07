# vote.py
import logging
import pandas as pd
import os
from collections import Counter

PREDICTIONS_DIR = 'data/predictions'

def read_predictions(prefix, dataset_type):
    files = [f for f in os.listdir(PREDICTIONS_DIR) if f.startswith(prefix) and dataset_type in f]
    if not files:
        logging.error(f"No prediction files found for {prefix} with type {dataset_type}.")
        return None
    predictions = []
    for file in files:
        path = os.path.join(PREDICTIONS_DIR, file)
        df = pd.read_csv(path)
        predictions.append(df[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']])
    combined_predictions = pd.concat(predictions, axis=0)
    return combined_predictions

def get_all_predictions():
    combined = [read_predictions(model, "combined") for model in ["rossler", "chua", "henon", "logistic", "lorenz96"]]
    pb = [read_predictions(model, "pb") for model in ["rossler", "chua", "henon", "logistic", "lorenz96"]]
    mb = [read_predictions(model, "mb") for model in ["rossler", "chua", "henon", "logistic", "lorenz96"]]
    return combined, pb, mb

def ensemble_voting(predictions):
    combined_predictions = []
    for dataset_predictions in predictions:
        if dataset_predictions is None or dataset_predictions.empty:
            logging.error("No valid predictions found for this dataset.")
            continue
        dataset_predictions = [pred for pred in dataset_predictions.values if not any(pd.isnull(pred))]
        all_predictions = pd.DataFrame(dataset_predictions, columns=['num1', 'num2', 'num3', 'num4', 'num5', 'numA'])
        most_common = [Counter(all_predictions[col]).most_common(1)[0][0] for col in all_predictions]
        combined_predictions.append(most_common)
    return combined_predictions

def run_voting_ensemble():
    logging.info("Running voting ensemble on predictions...")
    combined_predictions, pb_predictions, mb_predictions = get_all_predictions()

    logging.info("Top 5 Predictions for combined dataset:")
    combined_results = []
    for prediction_set in combined_predictions:
        if prediction_set is not None:
            combined_results.extend(ensemble_voting([prediction_set]))
    for i, prediction in enumerate(combined_results[:5]):
        logging.info(f"Prediction {i + 1}: {prediction}")

    logging.info("Top 5 Predictions for PB dataset:")
    pb_results = []
    for prediction_set in pb_predictions:
        if prediction_set is not None:
            pb_results.extend(ensemble_voting([prediction_set]))
    for i, prediction in enumerate(pb_results[:5]):
        logging.info(f"Prediction {i + 1}: {prediction}")

    logging.info("Top 5 Predictions for MB dataset:")
    mb_results = []
    for prediction_set in mb_predictions:
        if prediction_set is not None:
            mb_results.extend(ensemble_voting([prediction_set]))
    for i, prediction in enumerate(mb_results[:5]):
        logging.info(f"Prediction {i + 1}: {prediction}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_voting_ensemble()
