# bandit_main.py
import logging
import numpy as np
import pandas as pd
from bandit.epsilon_greedy import run_epsilon_greedy, EpsilonGreedy
from bandit.ucb import run_ucb, UCB
from bandit.thompson_sampling import run_thompson_sampling, ThompsonSampling
from bandit.reward_definition import get_reward_probabilities
from bandit.feature_engineering import prepare_features
from bandit.simulate import run_simulations
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_trained_bandit(file_path, bandit):
    np.save(file_path, bandit.values)

def flatten_rewards(rewards):
    if isinstance(rewards, list) and all(isinstance(r, list) for r in rewards):
        return [item for sublist in rewards for item in sublist]
    return rewards

def main():
    # Load and prepare the datasets
    data_combined = prepare_features(pd.read_csv('data/train_combined.csv'))
    data_pb = prepare_features(pd.read_csv('data/train_pb.csv'))
    data_mb = prepare_features(pd.read_csv('data/train_mb.csv'))
    
    # Get reward probabilities
    probabilities = get_reward_probabilities()
    
    logging.info("Reward Probabilities:")
    logging.info(probabilities)
    
    # Run ε-Greedy algorithm
    logging.info("Running ε-Greedy algorithm...")
    rewards_combined_eg = run_epsilon_greedy(data_combined, epsilon=0.1, n_arms=3)
    rewards_pb_eg = run_epsilon_greedy(data_pb, epsilon=0.1, n_arms=3)
    rewards_mb_eg = run_epsilon_greedy(data_mb, epsilon=0.1, n_arms=3)

    # Save ε-Greedy models
    os.makedirs('bandit/models', exist_ok=True)
    save_trained_bandit('bandit/models/bandit_combined_eg_values.npy', EpsilonGreedy(0.1, 3))
    save_trained_bandit('bandit/models/bandit_pb_eg_values.npy', EpsilonGreedy(0.1, 3))
    save_trained_bandit('bandit/models/bandit_mb_eg_values.npy', EpsilonGreedy(0.1, 3))
    
    logging.info(f"Total Reward for combined dataset (ε-Greedy): {sum(rewards_combined_eg)}")
    logging.info(f"Total Reward for PB dataset (ε-Greedy): {sum(rewards_pb_eg)}")
    logging.info(f"Total Reward for MB dataset (ε-Greedy): {sum(rewards_mb_eg)}")
    
    # Run UCB algorithm
    logging.info("Running UCB algorithm...")
    rewards_combined_ucb = run_ucb(data_combined, n_arms=3)
    rewards_pb_ucb = run_ucb(data_pb, n_arms=3)
    rewards_mb_ucb = run_ucb(data_mb, n_arms=3)

    # Save UCB models
    save_trained_bandit('bandit/models/bandit_combined_ucb_values.npy', UCB(3))
    save_trained_bandit('bandit/models/bandit_pb_ucb_values.npy', UCB(3))
    save_trained_bandit('bandit/models/bandit_mb_ucb_values.npy', UCB(3))
    
    logging.info(f"Total Reward for combined dataset (UCB): {sum(rewards_combined_ucb)}")
    logging.info(f"Total Reward for PB dataset (UCB): {sum(rewards_pb_ucb)}")
    logging.info(f"Total Reward for MB dataset (UCB): {sum(rewards_mb_ucb)}")
    
    # Run Thompson Sampling algorithm
    logging.info("Running Thompson Sampling algorithm...")
    rewards_combined_ts = run_thompson_sampling(data_combined, n_arms=3)
    rewards_pb_ts = run_thompson_sampling(data_pb, n_arms=3)
    rewards_mb_ts = run_thompson_sampling(data_mb, n_arms=3)

    # Save Thompson Sampling models
    save_trained_bandit('bandit/models/bandit_combined_ts_values.npy', ThompsonSampling(3))
    save_trained_bandit('bandit/models/bandit_pb_ts_values.npy', ThompsonSampling(3))
    save_trained_bandit('bandit/models/bandit_mb_ts_values.npy', ThompsonSampling(3))
    
    logging.info(f"Total Reward for combined dataset (Thompson Sampling): {sum(rewards_combined_ts)}")
    logging.info(f"Total Reward for PB dataset (Thompson Sampling): {sum(rewards_pb_ts)}")
    logging.info(f"Total Reward for MB dataset (Thompson Sampling): {sum(rewards_mb_ts)}")

    # Call the simulation script
    logging.info("Running simulation on all combinations...")
    run_simulations()

if __name__ == "__main__":
    main()

