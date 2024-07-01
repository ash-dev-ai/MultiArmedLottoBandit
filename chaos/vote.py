# vote.py

import logging
import pandas as pd
import os
from collections import Counter

PREDICTIONS_DIR = 'data/predictions'

class PredictionReader:
    def __init__(self, prefix, dataset_type):
        self.prefix = prefix
        self.dataset_type = dataset_type

    def read_predictions(self):
        files = [f for f in os.listdir(PREDICTIONS_DIR) if f.startswith(self.prefix) and self.dataset_type in f]
        if not files:
            logging.error(f"No prediction files found for {self.prefix} with type {self.dataset_type}.")
            return None
        predictions = []
        for file in files:
            path = os.path.join(PREDICTIONS_DIR, file)
            df = pd.read_csv(path)
            logging.info(f"Reading file: {file} with columns: {df.columns.tolist()}")
            if 'numSum' not in df.columns or 'totalSum' not in df.columns:
                logging.error(f"File {file} does not contain the required columns.")
                continue
            predictions.append(df[['numSum', 'totalSum']])
        if not predictions:
            logging.error(f"No valid prediction files found for {self.prefix} with type {self.dataset_type}.")
            return None
        combined_predictions = pd.concat(predictions, axis=0)
        return combined_predictions

class EnsembleVoting:
    def __init__(self):
        self.models = ["rossler", "chua", "henon", "logistic", "lorenz96"]

    def get_all_predictions(self):
        combined = [PredictionReader(model, "combined").read_predictions() for model in self.models]
        pb = [PredictionReader(model, "pb").read_predictions() for model in self.models]
        mb = [PredictionReader(model, "mb").read_predictions() for model in self.models]
        return combined, pb, mb

    def vote(self, predictions):
        combined_predictions = []
        for dataset_predictions in predictions:
            if dataset_predictions is None or dataset_predictions.empty:
                logging.error("No valid predictions found for this dataset.")
                continue
            dataset_predictions = [pred for pred in dataset_predictions.values if not any(pd.isnull(pred))]
            all_predictions = pd.DataFrame(dataset_predictions, columns=['numSum', 'totalSum'])
            most_common = [Counter(all_predictions[col]).most_common(1)[0][0] for col in all_predictions]
            combined_predictions.append(most_common)
        return combined_predictions

    def run_voting_ensemble(self):
        logging.info("Running voting ensemble on predictions...")
        combined_predictions, pb_predictions, mb_predictions = self.get_all_predictions()

        logging.info("Top 5 Predictions for combined dataset:")
        combined_results = []
        for prediction_set in combined_predictions:
            if prediction_set is not None:
                combined_results.extend(self.vote([prediction_set]))
        for i, prediction in enumerate(combined_results[:5]):
            logging.info(f"Prediction {i + 1}: {prediction}")

        logging.info("Top 5 Predictions for PB dataset:")
        pb_results = []
        for prediction_set in pb_predictions:
            if prediction_set is not None:
                pb_results.extend(self.vote([prediction_set]))
        for i, prediction in enumerate(pb_results[:5]):
            logging.info(f"Prediction {i + 1}: {prediction}")

        logging.info("Top 5 Predictions for MB dataset:")
        mb_results = []
        for prediction_set in mb_predictions:
            if prediction_set is not None:
                mb_results.extend(self.vote([prediction_set]))
        for i, prediction in enumerate(mb_results[:5]):
            logging.info(f"Prediction {i + 1}: {prediction}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ensemble = EnsembleVoting()
    ensemble.run_voting_ensemble()
