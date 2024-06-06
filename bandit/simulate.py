# simulate.py
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from bandit.epsilon_greedy import EpsilonGreedy
from bandit.ucb import UCB
from bandit.thompson_sampling import ThompsonSampling
from bandit.reward_definition import calculate_reward

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_trained_bandit(file_path, BanditClass, **kwargs):
    values = np.load(file_path)
    bandit = BanditClass(**kwargs)
    bandit.values = values
    return bandit

def process_batch(batch_df, bandit):
    rewards = []
    for _, row in batch_df.iterrows():
        chosen_arm = bandit.select_arm()
        reward = calculate_reward(row)
        bandit.update(chosen_arm, reward)
        rewards.append(reward)
    return rewards

def run_simulations():
    prediction_dir = 'data/predictions'
    prediction_files = [os.path.join(prediction_dir, f) for f in os.listdir(prediction_dir) if f.endswith('.csv')]
    today = datetime.today().strftime('%Y-%m-%d')

    for file_path in prediction_files:
        dataset_name = os.path.basename(file_path).split('_')[0]
        dataset_type = os.path.basename(file_path).split('_')[2]

        logging.info(f"Running simulation on {dataset_name} predictions for {dataset_type}...")

        predictions = pd.read_csv(file_path)

        # Load the trained models
        bandit_eg = load_trained_bandit(f'bandit/models/bandit_{dataset_name}_eg_values.npy', EpsilonGreedy, epsilon=0.1, n_arms=3)
        bandit_ucb = load_trained_bandit(f'bandit/models/bandit_{dataset_name}_ucb_values.npy', UCB, n_arms=3)
        bandit_ts = load_trained_bandit(f'bandit/models/bandit_{dataset_name}_ts_values.npy', ThompsonSampling, n_arms=3)

        results = []

        for _, row in predictions.iterrows():
            prediction_set = pd.DataFrame([{
                f'num{i}': row[f'num{i}'] for i in range(1, 6)
            }])
            prediction_set['numA'] = row['numA']
            reward = calculate_reward(row)

            rewards_eg = process_batch(prediction_set, bandit_eg)
            rewards_ucb = process_batch(prediction_set, bandit_ucb)
            rewards_ts = process_batch(prediction_set, bandit_ts)

            results.append({
                'combination': '-'.join(map(str, [row[f'num{i}'] for i in range(1, 6)] + [row['numA']])),
                'rewards_eg': rewards_eg[0] * reward,
                'rewards_ucb': rewards_ucb[0] * reward,
                'rewards_ts': rewards_ts[0] * reward
            })

        results_df = pd.DataFrame(results)
        results_dir = f'data/simulations/{today}'
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f'{dataset_name}_{dataset_type}_simulation_results_{today}.csv')

        results_df.to_csv(results_file, index=False)
        logging.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    run_simulations()
