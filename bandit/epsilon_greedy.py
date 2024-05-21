# epsilon_greedy.py
import numpy as np
import pandas as pd
from bandit.reward_definition import calculate_reward

class EpsilonGreedy:
    def __init__(self, epsilon, n_arms):
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.values)
        else:
            return np.random.randint(0, self.n_arms)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value

def run_epsilon_greedy(data, epsilon, n_arms):
    bandit = EpsilonGreedy(epsilon, n_arms)
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
    
    # Run epsilon-greedy on each dataset
    rewards_combined = run_epsilon_greedy(data_combined, epsilon=0.1, n_arms=3)
    rewards_pb = run_epsilon_greedy(data_pb, epsilon=0.1, n_arms=3)
    rewards_mb = run_epsilon_greedy(data_mb, epsilon=0.1, n_arms=3)
    
    print(f"Total Reward for combined dataset: {sum(rewards_combined)}")
    print(f"Total Reward for PB dataset: {sum(rewards_pb)}")
    print(f"Total Reward for MB dataset: {sum(rewards_mb)}")