# ppo.py

import logging
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class PPOAgent:
    def __init__(self, state_size, action_size, actor_lr=0.0001, critic_lr=0.0002, gamma=0.99, clip_epsilon=0.2, epochs=10):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs

        self.actor_model = self.build_actor()
        self.critic_model = self.build_critic()

        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr)

    def build_actor(self):
        """Build the neural network model for the Actor."""
        inputs = Input(shape=(self.state_size,))
        layer = Dense(64, activation='relu')(inputs)
        layer = Dense(64, activation='relu')(layer)
        outputs = Dense(self.action_size, activation='softmax')(layer)  # Output probabilities for actions
        return Model(inputs, outputs)

    def build_critic(self):
        """Build the neural network model for the Critic."""
        inputs = Input(shape=(self.state_size,))
        layer = Dense(64, activation='relu')(inputs)
        layer = Dense(64, activation='relu')(layer)
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

    def train_step(self, states, actions, advantages, returns, old_probs):
        """Perform a training step for the Actor and Critic using PPO update rule."""
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            # Compute the predicted probabilities and values
            new_probs = self.actor_model(states)
            new_probs = tf.gather(new_probs, actions, batch_dims=1, axis=1)
            values = self.critic_model(states)

            # Compute the ratios
            ratios = tf.exp(tf.math.log(new_probs) - tf.math.log(old_probs))

            # Compute the actor loss with clipping
            surrogate1 = ratios * advantages
            surrogate2 = tf.clip_by_value(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

            # Compute the critic loss
            critic_loss = tf.reduce_mean(tf.square(returns - values))

        # Compute gradients and apply to actor and critic models
        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        del tape  # Explicitly delete the tape to free resources

    def train(self, episodes, data, batch_size=32, output_dir='reinforce/reinforcement_results'):
        """Train the PPO agent."""
        logging.info("Training PPO agent...")
        data = self.preprocess_data(data)  # Filter columns
        rewards_log = []

        for episode in range(episodes):
            states, actions, rewards, next_states, dones, old_probs = self.collect_trajectories(data, batch_size)
            returns, advantages = self.compute_advantages(rewards, next_states, dones)

            for _ in range(self.epochs):
                self.train_step(states, actions, advantages, returns, old_probs)

            total_reward = np.sum(rewards)
            rewards_log.append(total_reward)
            logging.info(f"Episode {episode + 1}/{episodes} completed with total reward {total_reward}.")

        # Save the training rewards to a CSV file
        rewards_df = pd.DataFrame(rewards_log, columns=['Total Reward'])
        rewards_df.to_csv(os.path.join(output_dir, 'ppo_rewards.csv'), index_label='Episode')

        # Save the final policy (action probabilities for each state) to a CSV file
        final_policy = self.actor_model(data.values).numpy()
        final_policy_df = pd.DataFrame(final_policy, columns=[f'Action_{i}' for i in range(self.action_size)])
        final_policy_df.to_csv(os.path.join(output_dir, 'ppo_final_policy.csv'), index_label='State')

        return rewards_log  # Return the rewards log for main script use

    def collect_trajectories(self, data, batch_size):
        """Collect trajectories by interacting with the environment."""
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        old_probs = []

        for _ in range(batch_size):
            state = self.get_initial_state(data)
            action = self.choose_action(state)
            next_state, reward, done = self.take_action(state, action, data)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            old_probs.append(self.actor_model(state.reshape([1, self.state_size])).numpy().flatten()[action])

            if done:
                break

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), np.array(old_probs)

    def compute_advantages(self, rewards, next_states, dones):
        """Compute returns and advantages for each step in the trajectory."""
        returns = []
        advantages = []
        value_next = 0  # If the episode ends, the value of the next state is 0

        for reward, next_state, done in zip(reversed(rewards), reversed(next_states), reversed(dones)):
            if done:
                value_next = 0  # Reset value for terminal states
            value = reward + self.gamma * value_next
            advantage = reward + self.gamma * value_next - self.critic_model(next_state.reshape([1, self.state_size])).numpy()
            returns.append(value)
            advantages.append(advantage)
            value_next = value

        returns.reverse()
        advantages.reverse()
        return np.array(returns), np.array(advantages)

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
        state_numeric = np.asarray(state, dtype=np.float32)
        current_index = data.index[data.apply(lambda row: np.array_equal(row.values, state_numeric), axis=1)].tolist()
        if not current_index:
            current_index = [0]  # Default to the first state if not found
        next_index = (current_index[0] + action) % len(data)
        next_state = data.iloc[next_index].values
        reward = np.random.choice([0, 1])  # Random reward for demonstration
        done = next_index >= len(data) - 1
        return next_state, reward, done

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    state_size = 10  # Number of features used, including 'date'
    action_size = 10  # Adjust based on the number of possible actions
