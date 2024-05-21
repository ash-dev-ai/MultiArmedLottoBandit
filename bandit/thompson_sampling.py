# thompson_sampling.py
import numpy as np
import pandas as pd
from scipy.stats import beta
from bandit.reward_definition import calculate_reward

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms) + 1e-5  # Small positive values to avoid zero
        self.beta = np.ones(n_arms) + 1e-5  # Small positive values to avoid zero

    def select_arm(self):
        samples = [beta.rvs(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        return np.argmax(samples)

    def update(self, chosen_arm, reward):
        self.alpha[chosen_arm] = max(self.alpha[chosen_arm] + reward, 1e-5)
        self.beta[chosen_arm] = max(self.beta[chosen_arm] + 1 - reward, 1e-5)

def run_thompson_sampling(data, n_arms):
    bandit = ThompsonSampling(n_arms)
    rewards = []
    
    for index, row in data.iterrows():
        chosen_arm = bandit.select_arm()
        reward = calculate_reward(row)
        bandit.update(chosen_arm, reward)
        rewards.append(reward)
    
    return rewards

if __name__ == "__main__":
    # Load the datasets
    data_combined = pd.read_csv('../data/train_combined.csv')
    data_pb = pd.read_csv('../data/train_pb.csv')
    data_mb = pd.read_csv('../data/train_mb.csv')
    
    # Run Thompson Sampling on each dataset
    rewards_combined = run_thompson_sampling(data_combined, n_arms=3)
    rewards_pb = run_thompson_sampling(data_pb, n_arms=3)
    rewards_mb = run_thompson_sampling(data_mb, n_arms=3)
    
    print(f"Total Reward for combined dataset: {sum(rewards_combined)}")
    print(f"Total Reward for PB dataset: {sum(rewards_pb)}")
    print(f"Total Reward for MB dataset: {sum(rewards_mb)}")
