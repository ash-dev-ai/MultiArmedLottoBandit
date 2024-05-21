# bandit_main.py
import logging
import pandas as pd
from bandit.epsilon_greedy import run_epsilon_greedy
from bandit.ucb import run_ucb
from bandit.thompson_sampling import run_thompson_sampling
from bandit.reward_definition import get_reward_probabilities
from bandit.simulate import main as simulate_main

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load the datasets
    data_combined = pd.read_csv('data/train_combined.csv')
    data_pb = pd.read_csv('data/train_pb.csv')
    data_mb = pd.read_csv('data/train_mb.csv')
    
    # Get reward probabilities
    probabilities = get_reward_probabilities()
    
    logging.info("Reward Probabilities:")
    logging.info(probabilities)
    
    # Run ε-Greedy algorithm
    logging.info("Running ε-Greedy algorithm...")
    rewards_combined_eg = run_epsilon_greedy(data_combined, epsilon=0.1, n_arms=3)
    rewards_pb_eg = run_epsilon_greedy(data_pb, epsilon=0.1, n_arms=3)
    rewards_mb_eg = run_epsilon_greedy(data_mb, epsilon=0.1, n_arms=3)
    
    logging.info(f"Total Reward for combined dataset (ε-Greedy): {sum(rewards_combined_eg)}")
    logging.info(f"Total Reward for PB dataset (ε-Greedy): {sum(rewards_pb_eg)}")
    logging.info(f"Total Reward for MB dataset (ε-Greedy): {sum(rewards_mb_eg)}")
    
    # Run UCB algorithm
    logging.info("Running UCB algorithm...")
    rewards_combined_ucb = run_ucb(data_combined, n_arms=3)
    rewards_pb_ucb = run_ucb(data_pb, n_arms=3)
    rewards_mb_ucb = run_ucb(data_mb, n_arms=3)
    
    logging.info(f"Total Reward for combined dataset (UCB): {sum(rewards_combined_ucb)}")
    logging.info(f"Total Reward for PB dataset (UCB): {sum(rewards_pb_ucb)}")
    logging.info(f"Total Reward for MB dataset (UCB): {sum(rewards_mb_ucb)}")
    
    # Run Thompson Sampling algorithm
    logging.info("Running Thompson Sampling algorithm...")
    rewards_combined_ts = run_thompson_sampling(data_combined, n_arms=3)
    rewards_pb_ts = run_thompson_sampling(data_pb, n_arms=3)
    rewards_mb_ts = run_thompson_sampling(data_mb, n_arms=3)
    
    logging.info(f"Total Reward for combined dataset (Thompson Sampling): {sum(rewards_combined_ts)}")
    logging.info(f"Total Reward for PB dataset (Thompson Sampling): {sum(rewards_pb_ts)}")
    logging.info(f"Total Reward for MB dataset (Thompson Sampling): {sum(rewards_mb_ts)}")

    # Call the simulation script
    logging.info("Running simulation on all combinations...")
    simulate_main()

if __name__ == "__main__":
    main()


