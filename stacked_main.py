# stacked_main.py

import logging
from stacked.rnn_ensemble import main as rnn_main
from stacked.reservoir_ensemble import main as reservoir_main
from stacked.deep_learning_ensemble import main as dl_main
from stacked.meta_learner import main as meta_learner_main

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting the stacked ensemble process...")

    # Train and predict with RNN Ensemble
    logging.info("Running RNN Ensemble...")
    rnn_main()  # Calls the main function from rnn_ensemble.py

    # Train and predict with Reservoir Computing Ensemble
    logging.info("Running Reservoir Computing Ensemble...")
    reservoir_main()  # Calls the main function from reservoir_ensemble.py

    # Train and predict with Deep Learning Ensemble
    logging.info("Running Deep Learning Ensemble...")
    dl_main()  # Calls the main function from deep_learning_ensemble.py

    # Train and predict with Meta-Learner
    logging.info("Running Meta-Learner Ensemble...")
    meta_learner_main()  # Calls the main function from meta_learner.py

    logging.info("Stacked ensemble process completed.")

if __name__ == "__main__":
    main()
