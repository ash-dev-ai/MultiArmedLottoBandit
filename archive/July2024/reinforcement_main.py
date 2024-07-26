# reinforcement_main.py

import logging
import os
import pandas as pd

from reinforce.q_learning import QLearningModel
from reinforce.dqn import DQNModel
from reinforce.policy_gradient import PolicyGradient
from reinforce.actor_critic import ActorCritic
from reinforce.ppo import PPOAgent
from reinforce.mcts import MCTS
from reinforce.reinforced_predictions import PredictionAggregator
from reinforce.reinforce_rules import ReinforceRules

class ReinforcementLearningRunner:
    def __init__(self, input_dirs, output_dir, state_space_size, action_space_size, number_type):
        self.input_dirs = input_dirs
        self.output_dir = output_dir
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.number_type = number_type

        # Ensure output directory exists
        PredictionAggregator.ensure_directories_exist([self.output_dir])

    def load_data(self, file_path):
        """Load the CSV data file."""
        if not os.path.exists(file_path):
            logging.error(f"Data file {file_path} not found.")
            return None
        return pd.read_csv(file_path)

    def preprocess_data(self, data):
        """Preprocess data to include only specified columns."""
        columns_to_use = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']
        return data[columns_to_use]

    def save_results(self, file_path, results):
        """Save the results to a specified file."""
        with open(file_path, 'w') as file:
            for line in results:
                file.write(line + '\n')

    def run_q_learning(self, data, episodes=100):
        """Run the Q-Learning algorithm and save results."""
        q_learning = QLearningModel(self.state_space_size, self.action_space_size)
        rewards = q_learning.train(episodes, data)
        results = [f"Episode {i+1}: Total Reward: {reward}" for i, reward in enumerate(rewards)]
        self.save_results(os.path.join(self.output_dir, 'q_learning_results.txt'), results)

    def run_dqn(self, data, episodes=100, batch_size=45, max_steps=33):
        """Run the DQN algorithm and save results."""
        dqn = DQNModel(self.state_space_size, self.action_space_size, self.number_type)
        rewards = dqn.train(episodes, data, batch_size, max_steps)
        results = [f"Episode {i+1}: Total Reward: {reward}" for i, reward in enumerate(rewards)]
        self.save_results(os.path.join(self.output_dir, 'dqn_results.txt'), results)

    def run_policy_gradient(self, data, episodes=100, batch_size=45):
        """Run the Policy Gradient algorithm and save results."""
        policy_gradient = PolicyGradient(self.state_space_size, self.action_space_size, self.number_type)
        rewards = policy_gradient.train(episodes, data, batch_size)
        results = [f"Episode {i+1}: Total Reward: {reward}" for i, reward in enumerate(rewards)]
        self.save_results(os.path.join(self.output_dir, 'policy_gradient_results.txt'), results)

    def run_actor_critic(self, data, episodes=100, batch_size=45):
        """Run the Actor-Critic algorithm and save results."""
        actor_critic = ActorCritic(self.state_space_size, self.action_space_size, self.number_type)
        rewards = actor_critic.train(episodes, data, batch_size)
        results = [f"Episode {i+1}: Total Reward: {reward}" for i, reward in enumerate(rewards)]
        self.save_results(os.path.join(self.output_dir, 'actor_critic_results.txt'), results)

    def run_ppo(self, data, episodes=100, batch_size=45):
        """Run the PPO algorithm and save results."""
        ppo = PPOAgent(self.state_space_size, self.action_space_size, self.number_type)
        rewards = ppo.train(episodes, data, batch_size)
        results = [f"Episode {i+1}: Total Reward: {reward}" for i, reward in enumerate(rewards)]
        self.save_results(os.path.join(self.output_dir, 'ppo_results.txt'), results)

    def run_mcts(self, max_iterations=100):
        """Run the MCTS algorithm and save results."""
        mcts = MCTS(self.state_space_size, self.action_space_size, max_iterations)
        initial_state = []  # Starting with an empty state
        best_action_sequence = mcts.search(initial_state)
        result = f"Best action sequence from MCTS: {best_action_sequence}"
        self.save_results(os.path.join(self.output_dir, 'mcts_results.txt'), [result])
        logging.info(result)

    def run_all_models(self, data):
        """Run all models and save results."""
        logging.info("Running Q-Learning...")
        self.run_q_learning(data)

        logging.info("Running DQN...")
        self.run_dqn(data, episodes=100, batch_size=45, max_steps=33)

        logging.info("Running Policy Gradient...")
        self.run_policy_gradient(data)

        logging.info("Running Actor-Critic...")
        self.run_actor_critic(data)

        logging.info("Running PPO...")
        self.run_ppo(data)

        logging.info("Running MCTS...")
        self.run_mcts()

    def preprocess_and_aggregate_predictions(self):
        """Preprocess and aggregate predictions."""
        logging.info("Preprocessing predictions...")
        aggregator = PredictionAggregator(self.input_dirs[0])
        number_types = {
            'combined': {'num1-num5': 70, 'numA': 26},
            'pb': {'num1-num5': 69, 'numA': 26},
            'mb': {'num1-num5': 70, 'numA': 25}
        }

        for dataset_name, number_type in number_types.items():
            output_file = os.path.join(self.output_dir, f"{dataset_name}_final_predictions")
            aggregator.make_final_predictions_for_dataset(dataset_name, number_type, output_file, num_predictions=5)

            # Load and display the final predictions
            self.display_final_predictions(f"{output_file}.csv")

    def display_final_predictions(self, file_path):
        """Load and display the final predictions."""
        if os.path.exists(file_path):
            predictions = pd.read_csv(file_path, header=None)
            logging.info(f"Final predictions from {file_path}:\n{predictions}")
        else:
            logging.error(f"File {file_path} not found.")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the reinforcement learning process...")

    # Define file paths and directories
    data_combined_path = 'data/data_combined.csv'
    data_pb_path = 'data/data_pb.csv'
    data_mb_path = 'data/data_mb.csv'
    output_dir = 'reinforce/reinforcement_results'

    input_dirs = ['data/predictions', 'data/preprocessed_predictions']
    number_type = {'num1-num5': 69, 'numA': 26}  # Adjust for your specific dataset
    runner = ReinforcementLearningRunner(input_dirs, output_dir, state_space_size=10, action_space_size=10, number_type=number_type)

    # Load the datasets
    data_combined = runner.load_data(data_combined_path)
    data_pb = runner.load_data(data_pb_path)
    data_mb = runner.load_data(data_mb_path)

    if data_combined is None or data_pb is None or data_mb is None:
        logging.error("Failed to load one or more datasets.")
        return

    # Preprocess the data
    data_combined = runner.preprocess_data(data_combined)
    data_pb = runner.preprocess_data(data_pb)
    data_mb = runner.preprocess_data(data_mb)

    logging.info("Data combined shape: %s", data_combined.shape)
    logging.info("Data PB shape: %s", data_pb.shape)
    logging.info("Data MB shape: %s", data_mb.shape)

    # Run all reinforcement learning models
    runner.run_all_models(data_combined)

    # Preprocess and aggregate predictions
    runner.preprocess_and_aggregate_predictions()

    logging.info("Reinforcement learning process completed.")

if __name__ == "__main__":
    main()
