# evaluation.py
import numpy as np

def evaluate_algorithm(rewards):
    """Evaluate the performance of the bandit algorithm."""
    total_reward = np.sum(rewards)
    average_reward = np.mean(rewards)
    return total_reward, average_reward

if __name__ == "__main__":
    # Example rewards list for testing
    example_rewards = [10, 20, 30, 40, 50]
    
    # Evaluate the example rewards
    total_reward, average_reward = evaluate_algorithm(example_rewards)
    print(f"Total Reward: {total_reward}")
    print(f"Average Reward: {average_reward}")

