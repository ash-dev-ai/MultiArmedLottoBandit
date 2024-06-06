# vote.py
import logging
import pandas as pd
import os
from collections import Counter

PREDICTIONS_DIR = 'data/predictions'

def read_predictions(prefix):
    files = [f for f in os.listdir(PREDICTIONS_DIR) if f.startswith(prefix)]
    if not files:
        logging.error(f"No prediction files found for {prefix}.")
        return None
    predictions = []
    for file in files:
        path = os.path.join(PREDICTIONS_DIR, file)
        df = pd.read_csv(path)
        predictions.append(df[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']])
    combined_predictions = pd.concat(predictions, axis=0)
    return combined_predictions

def get_all_predictions():
    rossler_predictions = read_predictions("rossler")
    chua_predictions = read_predictions("chua")
    henon_predictions = read_predictions("henon")
    logistic_predictions = read_predictions("logistic")
    lorenz96_predictions = read_predictions("lorenz96")
    return rossler_predictions, chua_predictions, henon_predictions, logistic_predictions, lorenz96_predictions

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
    predictions = get_all_predictions()
    combined_predictions = []
    for prediction_set in predictions:
        if prediction_set is not None:
            combined_predictions.append(ensemble_voting([prediction_set]))
    logging.info("Top 5 Predictions:")
    for i, prediction in enumerate(combined_predictions):
        logging.info(f"Prediction {i + 1}: {prediction}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_voting_ensemble()
