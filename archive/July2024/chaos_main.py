# chaos_main.py
import logging
from chaos.rossler import run_rossler
from chaos.chua import run_chua
from chaos.henon import run_henon
from chaos.logistic import run_logistic
from chaos.lorenz96 import run_lorenz96
from chaos.vote import EnsembleVoting

def main():
    logging.info("Running Rössler Attractor predictions...")
    run_rossler(return_predictions=True)
    
    logging.info("Running Chua's Circuit predictions...")
    run_chua(return_predictions=True)
    
    logging.info("Running Henon's Map predictions...")
    run_henon(return_predictions=True)
    
    logging.info("Running Logistic Map predictions...")
    run_logistic(return_predictions=True)
    
    logging.info("Running Lorenz96 predictions...")
    run_lorenz96(return_predictions=True)
    
    logging.info("Running voting ensemble on predictions...")
    ensemble = EnsembleVoting()
    ensemble.run_voting_ensemble()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
