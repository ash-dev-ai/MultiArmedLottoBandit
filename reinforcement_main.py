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
from reinforce.reinforced_predictions import make_final_predictions_for_dataset, ensure_directories_exist

def ensure_directories_exist(dirs):
    """Ensure that the necessary directories exist."""
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def load_data(file_path):
    """Load the CSV data file."""
    if not os.path.exists(file_path):
        logging.error(f"Data file {file_path} not found.")
        return None
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess data to include only specified columns."""
    columns_to_use = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']
    return data[columns_to_use]

def save_results(file_path, results):
    """Save the results to a specified file."""
    with open(file_path, 'w') as file:
        for line in results:
            file.write(line + '\n')

def run_q_learning(data, state_space_size, action_space_size, episodes=100):
    """Run the Q-Learning algorithm and save results."""
    q_learning = QLearningModel(state_space_size, action_space_size)
    rewards = q_learning.train(episodes, data)
    results = [f"Episode {i+1}: Total Reward: {reward}" for i, reward in enumerate(rewards)]
    save_results('reinforce/reinforcement_results/q_learning_results.txt', results)

def run_dqn(data, state_size, action_size, episodes=100, batch_size=32, max_steps=10):
    """Run the DQN algorithm and save results."""
    dqn = DQNModel(state_size, action_size)
    rewards = dqn.train(episodes, data, batch_size, max_steps)
    results = [f"Episode {i+1}: Total Reward: {reward}" for i, reward in enumerate(rewards)]
    save_results('reinforce/reinforcement_results/dqn_results.txt', results)

def run_policy_gradient(data, state_size, action_size, episodes=100, batch_size=32):
    """Run the Policy Gradient algorithm and save results."""
    policy_gradient = PolicyGradient(state_size, action_size)
    rewards = policy_gradient.train(episodes, data, batch_size)
    results = [f"Episode {i+1}: Total Reward: {reward}" for i, reward in enumerate(rewards)]
    save_results('reinforce/reinforcement_results/policy_gradient_results.txt', results)

def run_actor_critic(data, state_size, action_size, episodes=100, batch_size=32):
    """Run the Actor-Critic algorithm and save results."""
    actor_critic = ActorCritic(state_size, action_size)
    rewards = actor_critic.train(episodes, data, batch_size)
    results = [f"Episode {i+1}: Total Reward: {reward}" for i, reward in enumerate(rewards)]
    save_results('reinforce/reinforcement_results/actor_critic_results.txt', results)

def run_ppo(data, state_size, action_size, episodes=100, batch_size=32):
    """Run the PPO algorithm and save results."""
    ppo = PPOAgent(state_size, action_size)
    rewards = ppo.train(episodes, data, batch_size)
    results = [f"Episode {i+1}: Total Reward: {reward}" for i, reward in enumerate(rewards)]
    save_results('reinforce/reinforcement_results/ppo_results.txt', results)

def run_mcts(state_size, action_size, max_iterations=100):
    """Run the MCTS algorithm and save results."""
    mcts = MCTS(state_size, action_size, max_iterations)
    initial_state = []  # Starting with an empty state
    best_action_sequence = mcts.search(initial_state)
    result = f"Best action sequence from MCTS: {best_action_sequence}"
    save_results('reinforce/reinforcement_results/mcts_results.txt', [result])
    logging.info(result)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the reinforcement learning process...")

    # Define file paths and directories
    data_combined_path = 'data/data_combined.csv'
    data_pb_path = 'data/data_pb.csv'
    data_mb_path = 'data/data_mb.csv'
    output_dir = 'reinforce/reinforcement_results'
    
    ensure_directories_exist([output_dir])

    # Load the datasets
    data_combined = load_data(data_combined_path)
    data_pb = load_data(data_pb_path)
    data_mb = load_data(data_mb_path)

    if data_combined is None or data_pb is None or data_mb is None:
        logging.error("Failed to load one or more datasets.")
        return

    # Preprocess the data
    data_combined = preprocess_data(data_combined)
    data_pb = preprocess_data(data_pb)
    data_mb = preprocess_data(data_mb)

    logging.info(f"Data combined shape: {data_combined.shape}")
    logging.info(f"Data PB shape: {data_pb.shape}")
    logging.info(f"Data MB shape: {data_mb.shape}")

    # Define state and action space sizes
    state_space_size = 10  # Updated to match the number of columns used for the state
    action_space_size = 70  # Define the largest number range for combined dataset (1-70)

    # Run each reinforcement learning method and save results
    logging.info("Running Q-Learning...")
    run_q_learning(data_combined, state_space_size, action_space_size)

    logging.info("Running DQN...")
    run_dqn(data_combined, state_space_size, action_space_size, episodes=100, batch_size=32, max_steps=10)

    logging.info("Running Policy Gradient...")
    run_policy_gradient(data_combined, state_space_size, action_space_size)

    logging.info("Running Actor-Critic...")
    run_actor_critic(data_combined, state_space_size, action_space_size)

    logging.info("Running PPO...")
    run_ppo(data_combined, state_space_size, action_space_size)

    logging.info("Running MCTS...")
    run_mcts(state_space_size, action_space_size)

    # Generate predictions for the next draw
    logging.info("Generating predictions for the next draw...")
    number_types = {
        'combined': {'num1-num5': 70, 'numA': 26},
        'pb': {'num1-num5': 69, 'numA': 26},
        'mb': {'num1-num5': 70, 'numA': 25}
    }
    for dataset_name, number_type in number_types.items():
        dataset_path = os.path.join('data', f'{dataset_name}.csv')
        data = load_data(dataset_path)
        if data is not None:
            data = preprocess_data(data)
            output_file = os.path.join(output_dir, f"{dataset_name}_final_predictions.csv")
            make_final_predictions_for_dataset(dataset_name, number_type, output_file)
        else:
            logging.error(f"Failed to load data for {dataset_name}. Skipping predictions for this dataset.")

    logging.info("Reinforcement learning process completed.")

if __name__ == "__main__":
    main()
