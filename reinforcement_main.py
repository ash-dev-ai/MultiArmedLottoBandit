# reinforcement_main.py

import logging
from reinforce.q_learning import QLearningModel
from reinforce.dqn import DQNModel
from reinforce.policy_gradient import PolicyGradientModel
from reinforce.actor_critic import ActorCriticModel
from reinforce.ppo import PPOModel
from reinforce.mcts import MCTSModel

def run_q_learning():
    logging.info("Running Q-Learning model...")
    q_learning = QLearningModel()
    q_learning.train()
    q_learning.predict()

def run_dqn():
    logging.info("Running DQN model...")
    dqn = DQNModel()
    dqn.train()
    dqn.predict()

def run_policy_gradient():
    logging.info("Running Policy Gradient model...")
    pg = PolicyGradientModel()
    pg.train()
    pg.predict()

def run_actor_critic():
    logging.info("Running Actor-Critic model...")
    ac = ActorCriticModel()
    ac.train()
    ac.predict()

def run_ppo():
    logging.info("Running PPO model...")
    ppo = PPOModel()
    ppo.train()
    ppo.predict()

def run_mcts():
    logging.info("Running MCTS model...")
    mcts = MCTSModel()
    mcts.train()
    mcts.predict()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting reinforcement learning process...")

    # Example: Running Q-Learning
    run_q_learning()
    # Uncomment the following lines to run other models
    # run_dqn()
    # run_policy_gradient()
    # run_actor_critic()
    # run_ppo()
    # run_mcts()

    logging.info("Reinforcement learning process completed.")

if __name__ == "__main__":
    main()


