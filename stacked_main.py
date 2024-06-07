# stacked_main.py
import logging
from stacked.load_data import load_datasets
from stacked.rnn_ensemble import train_rnn_ensemble, evaluate_rnn_ensemble
from stacked.reservoir_ensemble import train_reservoir_ensemble, evaluate_reservoir_ensemble
from stacked.deep_learning_ensemble import train_deep_learning_ensemble, evaluate_deep_learning_ensemble
from stacked.meta_learner import train_meta_learner, evaluate_meta_learner
from stacked.save_results import save_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Loading datasets...")
    datasets = load_datasets()
    
    # Process each dataset (combined, pb, mb)
    for dataset_type, (train_data, val_data, test_data) in datasets.items():
        logging.info(f"Processing {dataset_type} dataset...")

        # Train and evaluate RNN ensemble
        rnn_models = train_rnn_ensemble(train_data, val_data, dataset_type)
        rnn_predictions = evaluate_rnn_ensemble(rnn_models, test_data)

        # Train and evaluate Reservoir ensemble
        reservoir_models = train_reservoir_ensemble(train_data, val_data, dataset_type)
        reservoir_predictions = evaluate_reservoir_ensemble(reservoir_models, test_data)

        # Train and evaluate Deep Learning ensemble
        deep_learning_models = train_deep_learning_ensemble(train_data, val_data, dataset_type)
        deep_learning_predictions = evaluate_deep_learning_ensemble(deep_learning_models, test_data)

        # Train and evaluate Meta Learner (Transformer)
        combined_predictions = {
            'rnn': rnn_predictions,
            'reservoir': reservoir_predictions,
            'deep_learning': deep_learning_predictions
        }
        meta_learner = train_meta_learner(combined_predictions, val_data, dataset_type)
        final_predictions = evaluate_meta_learner(meta_learner, combined_predictions)

        logging.info("Saving predictions...")
        save_predictions(final_predictions, dataset_type)

    logging.info("Stacked model process complete.")

if __name__ == "__main__":
    main()


