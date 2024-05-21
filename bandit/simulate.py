# simulate.py
import logging
import pandas as pd
from bandit.epsilon_greedy import run_epsilon_greedy
from bandit.ucb import run_ucb
from bandit.thompson_sampling import run_thompson_sampling
from init.database import Database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_batch(batch, n_arms):
    rewards_eg = run_epsilon_greedy(batch, epsilon=0.1, n_arms=n_arms)
    rewards_ucb = run_ucb(batch, n_arms=n_arms)
    rewards_ts = run_thompson_sampling(batch, n_arms=n_arms)
    return rewards_eg, rewards_ucb, rewards_ts

def main():
    db = Database('data/numbers.db')
    batch_size = 100000
    
    results = []
    try:
        all_combinations = db.fetch_all_combinations(batch_size=batch_size)
        
        for batch in all_combinations:
            n_arms = len(batch)
            rewards_eg, rewards_ucb, rewards_ts = process_batch(batch, n_arms)
            
            batch_results = pd.DataFrame({
                'combination': batch.apply(lambda x: '-'.join(map(str, x)), axis=1),
                'rewards_eg': rewards_eg,
                'rewards_ucb': rewards_ucb,
                'rewards_ts': rewards_ts
            })
            results.append(batch_results)
    finally:
        db.close_connection()
    
    combined_results = pd.concat(results, ignore_index=True)
    combined_results.to_csv('data/model_results.csv', index=False)
    
    logging.info("Simulation complete. Results saved to data/model_results.csv")

if __name__ == "__main__":
    main()
