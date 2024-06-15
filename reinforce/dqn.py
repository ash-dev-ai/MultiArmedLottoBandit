# dqn.py

import logging
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNModel:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration
        self.model = self.build_model()

    def build_model(self):
        """Build the neural network model for DQN."""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """Choose an action based on the epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Exploit

    def replay(self, batch_size):
        """Experience replay to train the network."""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes, data, batch_size=32, max_steps=100):
        """Train the DQN model."""
        logging.info("Training DQN model...")
        data = self.preprocess_data(data)  # Filter columns
        rewards_per_episode = []

        for episode in range(episodes):
            state = self.get_initial_state(data)
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0
            done = False
            step = 0

            while not done and step < max_steps:
                action = self.choose_action(state)
                next_state, reward, done = self.take_action(step, action, data)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step += 1  # Increment step

                # Logging internal state details
                logging.info(f"Episode: {episode + 1}, Step: {step}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}, Done: {done}")

                if done:
                    logging.info(f"Episode {episode + 1} finished after {step} steps with total reward {total_reward}")
                    break

            rewards_per_episode.append(total_reward)

            if len(self.memory) > batch_size:
                self.replay(batch_size)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        return rewards_per_episode  # Return rewards for each episode

    def predict(self, state):
        """Predict the best action for a given state."""
        state = np.reshape(state, [1, self.state_size])
        return np.argmax(self.model.predict(state)[0])

    def get_initial_state(self, data):
        """Get the initial state for training."""
        return data.iloc[0].values  # Example: first row of data

    def preprocess_data(self, data):
        """Preprocess data to include only specified columns."""
        columns_to_use = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']
        data = data[columns_to_use]
        return data

    def take_action(self, step, action, data):
        """Take an action and return the next state, reward, and if the episode is done."""
        current_index = step % len(data)  # Use step for index
        next_index = (current_index + action) % len(data)
        next_state = data.iloc[next_index].values
        reward = np.random.choice([0, 1])  # Example: random reward
        done = next_index >= len(data) - 1  # Episode ends when reaching the end of data
        return next_state, reward, done

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    state_size = 10  # Adjusted to the number of features used
    action_size = 10  # Adjust based on the number of possible actions
    dqn = DQNModel(state_size, action_size)

    # Load and preprocess your data
    data = pd.read_csv('data/data_combined.csv')  # Replace with actual path to your data

    rewards = dqn.train(episodes=10, data=data, max_steps=50)  # Use smaller max_steps for testing
    logging.info(f"Training completed with rewards: {rewards}")

    # Example prediction
    state = data.iloc[0].values  # Example state
    action = dqn.predict(state)
    logging.info(f"Predicted action for state {state}: {action}")
