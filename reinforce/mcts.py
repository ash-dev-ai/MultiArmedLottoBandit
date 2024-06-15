# mcts.py

import logging
import numpy as np
import math
import random
import os
import pandas as pd
from bandit.reward_definition import calculate_reward

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def best_child(self, exploration_param=1.4):
        """Select the child node with the highest UCT value."""
        choices_weights = [
            (child.value / child.visits) + exploration_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self, action):
        """Expand the current node by adding a new child for the given action."""
        child_state = self.state + [action]
        child_node = Node(state=child_state, parent=self)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        """Update the node's value and visits with the given reward."""
        self.visits += 1
        self.value += reward

    def most_visited_child(self):
        """Return the child node that has been visited the most times."""
        return max(self.children, key=lambda child: child.visits)


class MCTS:
    def __init__(self, state_size, action_size, max_iterations=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.max_iterations = max_iterations

    def search(self, initial_state, output_dir='reinforce/reinforcement_results'):
        root = Node(state=initial_state)
        best_sequences = []

        for iteration in range(self.max_iterations):
            node = self.tree_policy(root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
            
            # Record the best action sequence and its reward at each iteration
            best_child = root.most_visited_child()
            best_sequences.append((iteration + 1, best_child.state, best_child.value))

        # Save the best action sequences and their rewards to a CSV file
        results_df = pd.DataFrame(best_sequences, columns=['Iteration', 'Best Action Sequence', 'Reward'])
        results_df.to_csv(os.path.join(output_dir, 'mcts_best_sequences.csv'), index=False)

        return root.most_visited_child().state

    def tree_policy(self, node):
        """Apply tree policy (selection and expansion) to reach a leaf node."""
        while len(node.state) < self.state_size:
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child()
        return node

    def expand(self, node):
        """Expand the node by adding a new child node."""
        available_actions = self.get_available_actions(node.state)
        action = random.choice(available_actions)
        return node.expand(action)

    def simulate(self, node):
        """Run a simulation from the given node's state to estimate a reward."""
        current_state = node.state.copy()
        total_reward = 0
        while len(current_state) < self.state_size:
            action = random.choice(self.get_available_actions(current_state))
            current_state.append(action)
            total_reward += self.estimate_reward(current_state)
        return total_reward

    def backpropagate(self, node, reward):
        """Backpropagate the simulation result up the tree to update node values."""
        while node is not None:
            node.update(reward)
            node = node.parent

    def get_available_actions(self, state):
        """Return a list of available actions given the current state."""
        return list(range(1, self.action_size + 1))

    def estimate_reward(self, state):
        """Estimate the reward for the given state using the reward definition logic."""
        # Convert the state to a dictionary for compatibility with `calculate_reward`
        state_dict = {
            'num1': state[0] if len(state) > 0 else 0,
            'num2': state[1] if len(state) > 1 else 0,
            'num3': state[2] if len(state) > 2 else 0,
            'num4': state[3] if len(state) > 3 else 0,
            'num5': state[4] if len(state) > 4 else 0,
            'numA': state[5] if len(state) > 5 else 0,
        }

        # Log the state dictionary for debugging purposes
        logging.debug(f"State dictionary: {state_dict}")

        # Calculate the reward using the imported `calculate_reward` function
        reward = calculate_reward(state_dict)
        return reward

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    state_size = 6  # The length of the state (number of lottery numbers to predict)
    action_size = 69  # Number of possible actions (e.g., lottery numbers from 1 to 69)
    mcts = MCTS(state_size, action_size)

    initial_state = []  # Starting with an empty state
    best_action_sequence = mcts.search(initial_state)
    logging.info(f"Best action sequence: {best_action_sequence}")
