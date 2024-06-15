# policy_gradient.py

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class PolicyGradient:
    def __init__(self, state_size, action_size, learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def build_model(self):
        """Build the neural network model for policy gradient."""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))  # Output probabilities for actions
        return model

    def choose_action(self, state):
        """Choose an action based on the policy network's output probabilities."""
        state = state.reshape([1, self.state_size])

        # Check for NaN in state and handle it
        if np.any(np.isnan(state)):
            logging.error("State contains NaN values. State: {}".format(state))
            state = np.nan_to_num(state)  # Replace NaNs with zero

        action_probs = self.model(state).numpy().flatten()

        # Check for NaN in action probabilities
        if np.any(np.isnan(action_probs)):
            logging.error("Action probabilities contain NaN values before normalization. Action probabilities: {}".format(action_probs))
            action_probs = np.nan_to_num(action_probs)  # Replace NaNs with zero

        # Log action_probs for debugging
        logging.debug(f"Action probabilities before normalization: {action_probs}")

        # Avoid division by zero in normalization
        sum_probs = np.sum(action_probs)
        if sum_probs == 0:
            logging.error("Sum of action probabilities is zero, setting uniform distribution.")
            action_probs = np.ones(self.action_size) / self.action_size  # Uniform probabilities
        else:
            action_probs = action_probs / sum_probs  # Normalize to ensure they sum to 1

        # Log normalized action_probs for debugging
        logging.debug(f"Action probabilities after normalization: {action_probs}")

        return np.random.choice(self.action_size, p=action_probs)

    def discount_rewards(self, rewards, gamma=0.95):
        """Compute the discounted reward for each time step."""
        discounted_rewards = np.zeros_like(rewards)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * gamma + rewards[t]
            discounted_rewards[t] = cumulative
        return discounted_rewards

    def train_step(self, states, actions, rewards):
        """Perform a training step using the REINFORCE algorithm."""
        discounted_rewards = self.discount_rewards(rewards)

        with tf.GradientTape() as tape:
            action_probs = self.model(states)
            action_mask = tf.one_hot(actions, self.action_size)
            log_probs = tf.reduce_sum(action_mask * tf.math.log(action_probs + 1e-10), axis=1)  # Add epsilon to avoid log(0)
            loss = -tf.reduce_mean(log_probs * discounted_rewards)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train(self, episodes, data, batch_size=32, output_dir='reinforce/reinforcement_results'):
        """Train the policy gradient model."""
        logging.info("Training Policy Gradient model...")
        data = self.preprocess_data(data)  # Filter columns
        rewards_per_episode = []

        for episode in range(episodes):
            states = []
            actions = []
            rewards = []
            state = self.get_initial_state(data)

            while True:
                action = self.choose_action(state)
                next_state, reward, done = self.take_action(state, action, data)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state

                if done or len(rewards) >= batch_size:
                    states = np.vstack(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    self.train_step(states, actions, rewards)
                    break

            total_reward = np.sum(rewards)
            rewards_per_episode.append(total_reward)
            logging.info(f"Episode {episode + 1}: Total Reward: {total_reward}")

        # Save the rewards for each episode to a CSV file
        rewards_df = pd.DataFrame({'Episode': range(1, episodes + 1), 'Total Reward': rewards_per_episode})
        rewards_df.to_csv(os.path.join(output_dir, 'policy_gradient_rewards.csv'), index=False)

        return rewards_per_episode

    def predict(self, state):
        """Predict the best action for a given state."""
        state = state.reshape([1, self.state_size])
        action_probs = self.model(state).numpy().flatten()
        return np.argmax(action_probs)

    def get_initial_state(self, data):
        """Get the initial state for training."""
        return data.iloc[0].values  # Example: first row of data

    def preprocess_data(self, data):
        """Preprocess data to include only specified columns."""
        columns_to_use = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']
        return data[columns_to_use]

    def take_action(self, state, action, data):
        """Take an action and return the next state, reward, and if the episode is done."""
        state_index = data.index[data.apply(lambda row: np.array_equal(row.values, state), axis=1)].tolist()
        if not state_index:
            state_index = [0]  # Default to the first state if not found
        current_index = state_index[0]
        next_index = (current_index + action) % len(data)
        next_state = data.iloc[next_index].values
        reward = np.random.choice([0, 1])  # Random reward for demonstration
        done = next_index >= len(data) - 1
        return next_state, reward, done

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    state_size = 10  # Number of features used including 'date'
    action_size = 10  # Adjust based on the number of possible actions
    policy_gradient = PolicyGradient(state_size, action_size)

    # Load and preprocess your data
    data = pd.read_csv('data/data_combined.csv')  # Replace with actual path to your data

    rewards = policy_gradient.train(episodes=100, data=data)
    logging.info(f"Training completed with rewards: {rewards}")

    # Example prediction
    state = data.iloc[0].values  # Example state including 'date'
    action = policy_gradient.predict(state)
    logging.info(f"Predicted action for state {state}: {action}")
