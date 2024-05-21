# reward_definition.py
import pandas as pd

# Prize structure with corresponding reward amounts
PRIZE_STRUCTURE = {
    "5+numA": 20000000,
    "5": 1000000,
    "4+numA": 10000,
    "4": 500,
    "3+numA": 200,
    "3": 10,
    "2+numA": 10,
    "1+numA": 4,
    "numA": 2,
    "0": 0
}

# Probabilities of each reward
REWARD_PROBABILITIES = {
    "5+numA": 1 / 302575350,
    "5": 1 / 12607306,
    "4+numA": 1 / 931001,
    "4": 1 / 38792,
    "3+numA": 1 / 14547,
    "3": 1 / 606,
    "2+numA": 1 / 693,
    "1+numA": 1 / 89,
    "numA": 1 / 37
}

def calculate_reward(row):
    """Calculate the reward based on the row data."""
    num_matches = sum(1 for i in range(1, 6) if str(row[f'num{i}']) in [str(row[f'num{j}']) for j in range(1, 6)])
    has_numA = str(row['numA']) in [str(row[f'num{i}']) for i in range(1, 6)]
    
    if num_matches == 5 and has_numA:
        return 20000000
    elif num_matches == 5:
        return 1000000
    elif num_matches == 4 and has_numA:
        return 10000
    elif num_matches == 4:
        return 500
    elif num_matches == 3 and has_numA:
        return 200
    elif num_matches == 3:
        return 10
    elif num_matches == 2 and has_numA:
        return 10
    elif num_matches == 1 and has_numA:
        return 4
    elif has_numA:
        return 2
    else:
        return 0

def get_reward_probabilities():
    """Return the predefined probabilities of each reward."""
    return REWARD_PROBABILITIES

if __name__ == "__main__":
    # Print reward probabilities
    print("Reward Probabilities:")
    print(get_reward_probabilities())
