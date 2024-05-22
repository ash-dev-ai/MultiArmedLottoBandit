import itertools
from Data.Datasets import org_mm, org_pb, full_mm, full_pb, new_mm, new_pb
import time

# Record the start time
start_time = time.process_time()


# Define the mm_datasets and pb_datasets
mm_datasets = [org_mm, full_mm, new_mm]
pb_datasets = [org_pb, full_pb, new_pb]

# Define all_datasets
all_datasets = [mm_datasets, pb_datasets]

# Define the column names
columns = ['d', 'm', 'r', 'w1', 'w2', 'w3', 'w4', 'w5']

# Helper function to create all possible groups with given number of 'w' columns and 'r'
def create_group(group_name, w_count, has_r=False):
    w_columns = [f'w{i}' for i in range(1, 6)]
    combinations = []
    for w_combination in itertools.combinations(w_columns, w_count):
        combination = list(w_combination)
        if has_r:
            combination.append('r')
        combinations.append(combination)
    return combinations

# Helper function to create labels
def create_label(group_name, w_count, has_r=False):
    label = f"{w_count}w"
    if has_r:
        label += "+r"
    return label

# Define the groups and labels for all_datasets
groups_labels = {
    "mm_datasets": {
        create_label("1w+r", 1, has_r=True): create_group("1w+r", 1, has_r=True),
        create_label("2w+r", 2, has_r=True): create_group("2w+r", 2, has_r=True),
        create_label("3w", 3): create_group("3w", 3),
        create_label("3w+r", 3, has_r=True): create_group("3w+r", 3, has_r=True),
        create_label("4w", 4): create_group("4w", 4),
        create_label("4w+r", 4, has_r=True): create_group("4w+r", 4, has_r=True),
        create_label("5w", 5): create_group("5w", 5),
        create_label("5w+r", 5, has_r=True): create_group("5w+r", 5, has_r=True),
    },
    "pb_datasets": {
        create_label("1w+r", 1, has_r=True): create_group("1w+r", 1, has_r=True),
        create_label("2w+r", 2, has_r=True): create_group("2w+r", 2, has_r=True),
        create_label("3w", 3): create_group("3w", 3),
        create_label("3w+r", 3, has_r=True): create_group("3w+r", 3, has_r=True),
        create_label("4w", 4): create_group("4w", 4),
        create_label("4w+r", 4, has_r=True): create_group("4w+r", 4, has_r=True),
        create_label("5w", 5): create_group("5w", 5),
        create_label("5w+r", 5, has_r=True): create_group("5w+r", 5, has_r=True),
    }
}

# Print the groups and labels for all_datasets
for dataset_name, groups in groups_labels.items():
    print(f"{dataset_name} groups:")
    for label, group in groups.items():
        print(f"{label} {group}")
        
# Calculate and print the elapsed time
end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time:.4f} seconds")
