# ucb.py
import numpy as np
import pandas as pd
from math import sqrt, log
from bandit.reward_definition import calculate_reward

class UCB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0

    def select_arm(self):
        if 0 in self.counts:
            return np.argmin(self.counts)
        ucb_values = self.values + np.sqrt(2 * log(self.total_counts) / self.counts)
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.total_counts += 1
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value

def run_ucb(data, n_arms):
    bandit = UCB(n_arms)
    rewards = []
    
    for index, row in data.iterrows():
        chosen_arm = bandit.select_arm()
        reward = calculate_reward(row)
        bandit.update(chosen_arm, reward)
        rewards.append(reward)
    
    return rewards

if __name__ == "__main__":
    # Load the datasets
    data_combined = pd.read_csv('data/train_combined.csv')
    data_pb = pd.read_csv('data/train_pb.csv')
    data_mb = pd.read_csv('data/train_mb.csv')
    
    # Run UCB on each dataset
    rewards_combined = run_ucb(data_combined, n_arms=3)
    rewards_pb = run_ucb(data_pb, n_arms=3)
    rewards_mb = run_ucb(data_mb, n_arms=3)
    
    print(f"Total Reward for combined dataset: {sum(rewards_combined)}")
    print(f"Total Reward for PB dataset: {sum(rewards_pb)}")
    print(f"Total Reward for MB dataset: {sum(rewards_mb)}")