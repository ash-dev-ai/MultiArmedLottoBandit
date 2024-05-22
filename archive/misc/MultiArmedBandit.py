import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import datetime

# Record the start time
start_time = time.process_time()

# The import for datasets
from MachinePairs import mm, pb

# Convert datasets to Pandas DataFrames
mm = pd.DataFrame(data_mm, columns=["d", "w", "r", "m"])
pb = pd.DataFrame(data_pb, columns=["d", "w", "r", "m"])

# Split the "w" column into individual columns w1, w2, w3, w4, w5
mm[["w1", "w2", "w3", "w4", "w5"]] = pd.DataFrame(mm["w"].tolist(), index=mm.index)
pb[["w1", "w2", "w3", "w4", "w5"]] = pd.DataFrame(pb["w"].tolist(), index=pb.index)

com_mm = mm
com_pb = pb

class MultiArmedBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.bandit_data = {
            'MM': {
                'arms': ['5W&R', '5W', '4W+R', '4W', '3W+R', '3W', '2W+R', '1W+R', '0W+R'],
                'odds': [302575350, 12607306, 931001, 38792, 14547, 606, 693, 89, 37],
                'prizes': [20000000, 1000000, 10000, 500, 200, 10, 10, 4, 2]
            },
            'PB': {
                'arms': ['5W&R', '5W', '4W&R', '4W', '3W&R', '3W', '2W&R', '1W&R', '0W&R'],
                'odds': [292201338, 11688054, 913129, 36525, 14494, 580, 701, 92, 38],
                'prizes': [20000000, 1000000, 50000, 100, 100, 7, 7, 4, 4]
            }
        }
        self.machine_data = {}
        self.total_pulls = 0

    def initialize_machines(self):
        for machine in self.bandit_data.keys():
            self.machine_data[machine] = {
                'total_reward': 0,
                'pull_count': 0
            }

    def pull_arm(self, machine):
        arm_index = self.choose_arm(machine)
        reward = self.get_reward(machine, arm_index)
        self.update_machine_data(machine, arm_index, reward)
        self.total_pulls += 1

    def choose_arm(self, machine):
        arm_odds = self.bandit_data[machine]['odds']
        return random.choices(range(len(arm_odds)), arm_odds)[0]

    def get_reward(self, machine, arm_index):
        return self.bandit_data[machine]['prizes'][arm_index]

    def update_machine_data(self, machine, arm_index, reward):
        self.machine_data[machine]['total_reward'] += reward
        self.machine_data[machine]['pull_count'] += 1

    def get_machine_stats(self, machine):
        return self.machine_data[machine]['total_reward'], self.machine_data[machine]['pull_count']

    def get_total_pulls(self):
        return self.total_pulls

# Main function to run the simulation
def run_bandit_simulation(num_pulls):
    bandit = MultiArmedBandit(num_arms=9)
    bandit.initialize_machines()

    for _ in range(num_pulls):
        # Assuming the output comes from two machines: 'MM' and 'PB'
        # Replace this with actual output data from the 'mm' and 'pb' datasets
        machine = random.choice(['MM', 'PB'])
        bandit.pull_arm(machine)

    # Get statistics for both machines
    total_pulls = bandit.get_total_pulls()
    mm_total_reward, mm_pull_count = bandit.get_machine_stats('MM')
    pb_total_reward, pb_pull_count = bandit.get_machine_stats('PB')

    # Print the results
    print(f"Total Pulls: {total_pulls}")
    print(f"MM Machine - Total Reward: ${mm_total_reward}, Pull Count: {mm_pull_count}")
    print(f"PB Machine - Total Reward: ${pb_total_reward}, Pull Count: {pb_pull_count}")


# Run the simulation for 5000 pulls
run_bandit_simulation(num_pulls=5000)

# Record the end time
end_time = time.process_time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"\nStart time: {start_time}")
print(f"End time: {end_time}")
print(f"Elapsed time: {elapsed_time} seconds")
