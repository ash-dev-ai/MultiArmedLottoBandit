# vote.py
import logging
from collections import Counter
import numpy as np
from chaos.rossler import run_rossler
from chaos.chua import run_chua
from chaos.henon import run_henon
from chaos.logistic import run_logistic
from chaos.lorenz96 import run_lorenz96

def get_model_predictions():
    logging.info("Getting predictions from Rössler Attractor model...")
    rossler_predictions = run_rossler(return_predictions=True)
    
    logging.info("Getting predictions from Chua's Circuit model...")
    chua_predictions = run_chua(return_predictions=True)
    
    logging.info("Getting predictions from Henon's Map model...")
    henon_predictions = run_henon(return_predictions=True)
    
    logging.info("Getting predictions from Logistic Map model...")
    logistic_predictions = run_logistic(return_predictions=True)
    
    logging.info("Getting predictions from Lorenz96 model...")
    lorenz96_predictions = run_lorenz96(return_predictions=True)
    
    return rossler_predictions, chua_predictions, henon_predictions, logistic_predictions, lorenz96_predictions

def ensemble_voting(predictions):
    combined_predictions = []

    # Find the maximum length of predictions
    max_length = max(len(pred[0]) for pred in predictions)
    
    # Ensure all predictions have the same length by padding with None
    padded_predictions = []
    for pred in predictions:
        padded = [np.pad(p, (0, max_length - len(p)), 'constant', constant_values=None) for p in pred]
        padded_predictions.append(np.array(padded))
    
    for dataset_predictions in zip(*padded_predictions):
        dataset_predictions = [pred for pred in dataset_predictions if pred is not None]  # Remove None values
        all_predictions = np.vstack(dataset_predictions)
        most_common = [Counter(all_predictions[:, i]).most_common(1)[0][0] for i in range(all_predictions.shape[1])]
        combined_predictions.append(most_common)
    
    return combined_predictions

def run_voting_ensemble():
    predictions = get_model_predictions()
    combined_predictions = ensemble_voting(predictions)
    
    logging.info("Top 5 Predictions:")
    for i, prediction in enumerate(combined_predictions):
        logging.info(f"Prediction {i + 1}: {prediction}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_voting_ensemble()
