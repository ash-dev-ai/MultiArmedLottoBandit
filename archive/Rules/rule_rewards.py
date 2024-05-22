from Groups import groups
import time

# Define the rewards for mm_groups
MM_REWARDS = {
    'r': 2,
    '1w+r': 4,
    '2w+r': 10,
    '3w': 10,
    '3w+r': 200,
    '4w': 500,
    '4w+r': 10000,
    '5w': 1000000,
    '5w+r': 20000000
}

# Define the rewards for pb_groups
PB_REWARDS = {
    'r': 4,
    '1w+r': 4,
    '2w+r': 7,
    '3w': 7,
    '3w+r': 100,
    '4w': 100,
    '4w+r': 50000,
    '5w': 1000000,
    '5w+r': 20000000
}

# Record the start time
start_time = time.process_time()

# Create new dictionaries mm_groups and pb_groups and add the odds to them
mm_groups = {}
for group in groups:
    mm_groups[group] = {'odds': MM_REWARDS[group]}

pb_groups = {}
for group in groups:
    pb_groups[group] = {'odds': PB_REWARDS[group]}

# Calculate and print the elapsed time
end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time:.4f} seconds")
