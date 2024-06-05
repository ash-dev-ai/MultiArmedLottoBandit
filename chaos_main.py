# chaos_main.py
import logging
from chaos.rossler import run_rossler
from chaos.chua import run_chua
from chaos.henon import run_henon
from chaos.logistic import run_logistic
from chaos.lorenz96 import run_lorenz96
from chaos.vote import run_voting_ensemble

def main():
    logging.info("Running Rössler model...")
    run_rossler()
    
    logging.info("Running Chua model...")
    run_chua()
    
    logging.info("Running Henon model...")
    run_henon()
    
    logging.info("Running Logistic model...")
    run_logistic()
    
    logging.info("Running Lorenz96 model...")
    run_lorenz96()
    
    logging.info("Running ensemble voting for chaos theory models...")
    run_voting_ensemble()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
