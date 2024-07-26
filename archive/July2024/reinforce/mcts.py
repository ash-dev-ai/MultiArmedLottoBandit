# mcts.py

import logging
import numpy as np
import math
import random
import os
import pandas as pd
from bandit.reward_definition import calculate_reward
from reinforce.reinforce_rules import ReinforceRules

class Node:
    def __init__(self, state: list, parent: 'Node' = None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self) -> bool:
        return len(self.children) > 0

    def best_child(self, exploration_param: float = 1.4) -> 'Node':
        """Select the child node with the highest UCT value."""
        choices_weights = [
            (child.value / child.visits) + exploration_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self, action: int) -> 'Node':
        """Expand the current node by adding a new child for the given action."""
        child_state = self.state + [action]
        child_node = Node(state=child_state, parent=self)
        self.children.append(child_node)
        return child_node

    def update(self, reward: float):
        """Update the node's value and visits with the given reward."""
        self.visits += 1
        self.value += reward

    def most_visited_child(self) -> 'Node':
        """Return the child node that has been visited the most times."""
        return max(self.children, key=lambda child: child.visits)

class MCTS:
    def __init__(self, state_size: int, action_size: int, number_type: dict, max_iterations: int = 1000):
        self.state_size = state_size
        self.action_size = action_size
        self.number_type = number_type
        self.max_iterations = max_iterations
        self.rules = ReinforceRules(number_type)

    def search(self, initial_state: list, output_dir: str = 'reinforce/reinforcement_results') -> list:
        root = Node(state=initial_state)
        best_sequences = []

        for iteration in range(self.max_iterations):
            node = self.tree_policy(root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
            
            # Record the best action sequence and its reward at each iteration
            best_child = root.most_visited_child()
            best_sequences.append((iteration + 1, best_child.state, best_child.value))

            # Log progress every 100 iterations
            if (iteration + 1) % 100 == 0:
                logging.info(f"Iteration {iteration + 1}/{self.max_iterations} completed.")

        # Save the best action sequences and their rewards to a CSV file
        self.save_results(best_sequences, output_dir)
        return root.most_visited_child().state

    def tree_policy(self, node: Node) -> Node:
        """Apply tree policy (selection and expansion) to reach a leaf node."""
        while len(node.state) < self.state_size:
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child()
        return node

    def expand(self, node: Node) -> Node:
        """Expand the node by adding a new child node."""
        available_actions = self.get_available_actions(node.state)
        action = random.choice(available_actions)
        return node.expand(action)

    def simulate(self, node: Node) -> float:
        """Run a simulation from the given node's state to estimate a reward."""
        current_state = node.state.copy()
        total_reward = 0
        while len(current_state) < self.state_size:
            action = random.choice(self.get_available_actions(current_state))
            current_state.append(action)
            reward = self.estimate_reward(current_state)

            # Validate the generated prediction
            if len(current_state) == 6:  # Only validate if the state is fully formed
                prediction_dict = {
                    'num1': current_state[0],
                    'num2': current_state[1],
                    'num3': current_state[2],
                    'num4': current_state[3],
                    'num5': current_state[4],
                    'numA': current_state[5],
                    'numSum': np.sum(current_state[:5]),
                }
                historical_data = pd.read_csv('data/data_combined.csv')[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']]
                if not self.rules.validate_all_rules(prediction_dict, historical_data):
                    total_reward -= 1  # Penalty for invalid prediction

            total_reward += reward

        return total_reward

    def backpropagate(self, node: Node, reward: float):
        """Backpropagate the simulation result up the tree to update node values."""
        while node is not None:
            node.update(reward)
            node = node.parent

    def get_available_actions(self, state: list) -> list:
        """Return a list of available actions given the current state."""
        return list(range(1, self.action_size + 1))

    def estimate_reward(self, state: list) -> float:
        """Estimate the reward for the given state using the reward definition logic."""
        state_dict = {
            'num1': state[0] if len(state) > 0 else 0,
            'num2': state[1] if len(state) > 1 else 0,
            'num3': state[2] if len(state) > 2 else 0,
            'num4': state[3] if len(state) > 3 else 0,
            'num5': state[4] if len(state) > 4 else 0,
            'numA': state[5] if len(state) > 5 else 0,
        }

        logging.debug(f"State dictionary: {state_dict}")
        reward = calculate_reward(state_dict)
        return reward

    def save_results(self, best_sequences: list, output_dir: str):
        """Save the best action sequences and their rewards to a CSV file."""
        results_df = pd.DataFrame(best_sequences, columns=['Iteration', 'Best Action Sequence', 'Reward'])
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(output_dir, 'mcts_best_sequences.csv'), index=False)

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname=s - %(message=s')

    state_size = 6  # The length of the state (number of lottery numbers to predict)
    action_size = 69  # Number of possible actions (e.g., lottery numbers from 1 to 69)
    number_type = {'num1-num5': 69, 'numA': 26}  # Define number type here
    mcts = MCTS(state_size, action_size, number_type)

    initial_state = []  # Starting with an empty state
    best_action_sequence = mcts.search(initial_state)
    logging.info(f"Best action sequence: {best_action_sequence}")
