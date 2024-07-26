# actor_critic.py

import logging
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from reinforce.reinforce_rules import ReinforceRules

class ActorCritic:
    def __init__(self, state_size, action_size, number_type, actor_lr=0.001, critic_lr=0.005, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.number_type = number_type
        self.gamma = gamma
        self.rules = ReinforceRules(number_type)

        self.actor_model = self.build_actor()
        self.critic_model = self.build_critic()

        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr)

    def build_actor(self):
        """Build the neural network model for the Actor."""
        inputs = Input(shape=(self.state_size,))
        layer = Dense(70, activation='relu')(inputs)
        layer = Dense(70, activation='relu')(layer)
        outputs = Dense(self.action_size, activation='softmax')(layer)  # Output probabilities for actions
        return Model(inputs, outputs)

    def build_critic(self):
        """Build the neural network model for the Critic."""
        inputs = Input(shape=(self.state_size,))
        layer = Dense(70, activation='relu')(inputs)
        layer = Dense(70, activation='relu')(layer)
        outputs = Dense(1, activation='linear')(layer)  # Output a single value (state value)
        return Model(inputs, outputs)

    def choose_action(self, state):
        """Choose an action based on the policy network's output probabilities."""
        state = state.reshape([1, self.state_size])
        action_probs = self.actor_model(state).numpy().flatten()

        # Log action probabilities
        logging.debug(f"Action probabilities: {action_probs}")

        # Ensure probabilities sum to 1 to prevent errors in np.random.choice
        action_probs = action_probs / np.sum(action_probs)

        # Check for NaN in action probabilities and handle it
        if np.any(np.isnan(action_probs)):
            logging.error("Action probabilities contain NaN values. Action probabilities: {}".format(action_probs))
            action_probs = np.nan_to_num(action_probs)  # Replace NaNs with zero

        return np.random.choice(self.action_size, p=action_probs)

    def train_step(self, state, action, reward, next_state, done):
        """Perform a training step for both the Actor and Critic."""
        state = state.reshape([1, self.state_size])
        next_state = next_state.reshape([1, self.state_size])

        with tf.GradientTape(persistent=True) as tape:
            # Predict the value for the current and next states
            value = self.critic_model(state)
            next_value = self.critic_model(next_state)

            # Calculate the target and advantage
            target = reward + (1 - done) * self.gamma * next_value
            advantage = target - value

            # Calculate actor loss
            action_probs = self.actor_model(state)
            action_mask = tf.one_hot(action, self.action_size)
            log_probs = tf.reduce_sum(action_mask * tf.math.log(action_probs + 1e-10), axis=1)  # Add epsilon to avoid log(0)
            actor_loss = -tf.reduce_mean(log_probs * advantage)

            # Calculate critic loss
            critic_loss = tf.reduce_mean(tf.square(target - value))

        # Compute gradients and apply to actor and critic models
        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        del tape  # Explicitly delete the tape to free resources

    def train(self, episodes, data, batch_size=32, output_dir='reinforce/reinforcement_results'):
        """Train the Actor-Critic model."""
        logging.info("Training Actor-Critic model...")
        data = self.preprocess_data(data)  # Filter columns
        rewards_per_episode = []

        for episode in range(episodes):
            state = self.get_initial_state(data)
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0
            steps = 0

            while True:
                action = self.choose_action(state)
                next_state, reward, done = self.take_action(state, action, data)
                next_state_full = np.zeros(self.state_size)  # Ensure the next_state has the correct size
                next_state_full[:len(next_state)] = next_state  # Fill in the known values
                next_state_full = np.reshape(next_state_full, [1, self.state_size])
                self.train_step(state, action, reward, next_state_full, done)

                state = next_state_full
                total_reward += reward
                steps += 1

                if done or steps >= batch_size:
                    break

            rewards_per_episode.append(total_reward)
            logging.info(f"Episode {episode + 1}: Total Reward: {total_reward}")

        # Save the rewards for each episode to a CSV file
        rewards_df = pd.DataFrame({'Episode': range(1, episodes + 1), 'Total Reward': rewards_per_episode})
        rewards_df.to_csv(os.path.join(output_dir, 'actor_critic_rewards.csv'), index=False)

        return rewards_per_episode

    def predict(self, state):
        """Predict the best action for a given state."""
        state = state.reshape([1, self.state_size])
        action_probs = self.actor_model(state).numpy().flatten()
        return np.argmax(action_probs)

    def get_initial_state(self, data):
        """Get the initial state for training."""
        return data.iloc[0].values

    def preprocess_data(self, data):
        """Preprocess data to include only specified columns."""
        columns_to_use = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']
        return data[columns_to_use]

    def take_action(self, state, action, data):
        """Take an action and return the next state, reward, and if the episode is done."""
        # Convert the state to match the data's dtype for comparison
        state_numeric = np.asarray(state, dtype=np.float32)
        current_index = data.index[data.apply(lambda row: np.array_equal(row.values, state_numeric), axis=1)].tolist()
        if not current_index:
            current_index = [0]  # Default to the first state if not found
        next_index = (current_index[0] + action) % len(data)
        next_state = data.iloc[next_index][['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].values

        # Validate the generated prediction
        prediction_dict = {
            'num1': next_state[0],
            'num2': next_state[1],
            'num3': next_state[2],
            'num4': next_state[3],
            'num5': next_state[4],
            'numA': next_state[5],
            'numSum': np.sum(next_state[:5]),
        }
        historical_data = data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']]

        if self.rules.validate_all_rules(prediction_dict, historical_data):
            reward = 1  # Reward for valid prediction
        else:
            reward = -1  # Penalty for invalid prediction

        done = next_index >= len(data) - 1
        return next_state, reward, done

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname=s - %(message=s')

    state_size = 10  # Number of features used, including 'date'
    action_size = 10  # Adjust based on the number of possible actions
    number_type = {'num1-num5': 69, 'numA': 26}  # Define number type here
    actor_critic = ActorCritic(state_size, action_size, number_type)

    # Load and preprocess your data
    data = pd.read_csv('data/data_combined.csv')  # Replace with actual path to your data

    rewards = actor_critic.train(episodes=100, data=data)
    logging.info(f"Training completed with rewards: {rewards}")

    # Example prediction
    state = data.iloc[0].values  # Example state including 'date'
    action = actor_critic.predict(state)
    logging.info(f"Predicted action for state {state}: {action}")
