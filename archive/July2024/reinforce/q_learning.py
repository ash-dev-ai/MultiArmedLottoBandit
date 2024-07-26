# q_learning.py

import logging
import numpy as np
import pandas as pd
from reinforce.reinforce_rules import ReinforceRules

class QLearningModel:
    def __init__(self, state_space_size: int, action_space_size: int, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        self.rules = ReinforceRules({'num1-num5': 69, 'numA': 26})  # Initialize ReinforceRules with number type

    def choose_action(self, state_index: int) -> int:
        """Choose an action based on epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space_size)  # Explore
        else:
            return np.argmax(self.q_table[state_index])  # Exploit

    def update_q_table(self, state_index: int, action: int, reward: float, next_state_index: int):
        """Update the Q-table using the Q-learning update rule."""
        old_value = self.q_table[state_index, action]
        next_max = np.max(self.q_table[next_state_index])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state_index, action] = new_value

    def train(self, episodes: int, data: pd.DataFrame) -> list:
        logging.info("Training Q-Learning model...")
        data = self.preprocess_data(data)  # Filter columns
        rewards_per_episode = []

        for episode in range(episodes):
            state_index = self.get_initial_state()
            total_reward = 0

            while not self.is_terminal_state(state_index):
                action = self.choose_action(state_index)
                next_state_index, reward = self.take_action(state_index, action, data)

                prediction = data.iloc[next_state_index][['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].to_dict()
                prediction['numSum'] = sum(prediction[key] for key in ['num1', 'num2', 'num3', 'num4', 'num5'])
                historical_data = data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']]

                if not self.rules.validate_all_rules(prediction, historical_data):
                    reward = -1  # Penalty for invalid prediction

                self.update_q_table(state_index, action, reward, next_state_index)
                state_index = next_state_index
                total_reward += reward

            rewards_per_episode.append(total_reward)
            logging.info(f"Episode {episode + 1}: Total Reward: {total_reward}")

        return rewards_per_episode  # Return the rewards for each episode

    def predict(self, state_index: int) -> int:
        """Predict the best action for a given state based on the Q-table."""
        return np.argmax(self.q_table[state_index])

    def get_initial_state(self) -> int:
        """Get the initial state index for training."""
        return 0  # Starting with the first index

    def take_action(self, state_index: int, action: int, data: pd.DataFrame) -> tuple:
        """Take an action and return the next state index and reward."""
        next_state_index = (state_index + action) % self.state_space_size
        reward = np.random.choice([0, 1])  # Random reward for demonstration
        return next_state_index, reward

    def is_terminal_state(self, state_index: int) -> bool:
        """Check if the current state index is terminal."""
        return state_index >= self.state_space_size - 1

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data to include only specified columns without converting date."""
        columns_to_use = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']
        return data[columns_to_use]

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    state_space_size = 100  # Adjust based on your data
    action_space_size = 10  # Adjust based on your possible actions
    q_learning = QLearningModel(state_space_size, action_space_size)

    # Load and preprocess your data
    data = pd.read_csv('data/data_combined.csv')  # Replace with actual path to your data

    rewards = q_learning.train(episodes=1000, data=data)
    logging.info(f"Training completed with rewards: {rewards}")
    
    # Example of predicting next action from a given state index
    state_index = 0  # Example state index
    action = q_learning.predict(state_index)
    logging.info(f"Predicted action for state index {state_index}: {action}")
