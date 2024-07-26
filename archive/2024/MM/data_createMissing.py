import random
import numpy as np
import pandas as pd
import time

# Record the start time
start_time = time.process_time()

# The import for datasets
from data_import import data_mm, data_pb

# Convert datasets to Pandas DataFrames and convert 'r' column to integers
mm = pd.DataFrame(data_mm, columns=["d", "w", "r", "m"])
mm["r"] = mm["r"].astype(int)

pb = pd.DataFrame(data_pb, columns=["d", "w", "r", "m"])
pb["r"] = pb["r"].astype(int)

# Convert datetime to an integer representing days since the earliest entry in the d column
# Assuming the earliest entry is the minimum date in both datasets
earliest_date_mm = mm["d"].min()
earliest_date_pb = pb["d"].min()

def datetime_to_days_since_earliest(date, earliest_date):
    return (date - earliest_date).days + 1

mm["d"] = mm["d"].apply(lambda x: datetime_to_days_since_earliest(x, earliest_date_mm))
pb["d"] = pb["d"].apply(lambda x: datetime_to_days_since_earliest(x, earliest_date_pb))

# MM Configuration
earliest_date_mm_int = datetime_to_days_since_earliest(earliest_date_mm, earliest_date_mm)
MM_W_BALLS = list(range(1, 71))  # White balls labeled 1-70
MM_R_BALLS = list(range(1, 26))  # Red balls labeled 1-25

# PB Configuration
earliest_date_pb_int = datetime_to_days_since_earliest(earliest_date_pb, earliest_date_pb)
PB_W_BALLS = list(range(1, 70))  # White balls labeled 1-69
PB_R_BALLS = list(range(1, 27))  # Red balls labeled 1-26

# Function to create the gravity pick machine
def create_gravity_pick_machine(earliest_date_int, W_BALLS, R_BALLS):
    # Ensure R_BALLS only contains integers
    R_BALLS = [int(ball) for ball in R_BALLS]
    
    machine_PM = {}
    machine_MM = {}
    for date_int in range(earliest_date_int, earliest_date_int + 1000):
        W_machine_balls_PM = random.sample(W_BALLS, 5)
        R_machine_ball_PM = random.choice(R_BALLS)
        W_machine_balls_MM = random.sample(W_BALLS, 5)
        R_machine_ball_MM = random.choice(R_BALLS)
        machine_PM[date_int] = {"W": W_machine_balls_PM, "R": R_machine_ball_PM}
        machine_MM[date_int] = {"W": W_machine_balls_MM, "R": R_machine_ball_MM}
    return machine_PM, machine_MM

# Create MM gravity pick machines
MM_PM, MM_MM = create_gravity_pick_machine(earliest_date_mm_int, MM_W_BALLS, MM_R_BALLS)

# Create PB gravity pick machines
PB_PM, PB_MM = create_gravity_pick_machine(earliest_date_pb_int, PB_W_BALLS, PB_R_BALLS)

# Function to fill in the missing entries in the datasets using the gravity pick machines
def fill_missing_entries(df, machine_PM, machine_MM):
    new_entries = []
    all_dates = list(range(df["d"].min(), df["d"].max() + 1))
    available_dates_PM = list(machine_PM.keys())
    available_dates_MM = list(machine_MM.keys())
    for date_int in all_dates:
        if date_int not in df["d"].values:
            new_w_balls_PM = []
            new_r_ball_PM = None

            if date_int in available_dates_PM:
                new_w_balls_PM = machine_PM[date_int]["W"]
                new_r_ball_PM = machine_PM[date_int]["R"]

            new_w_balls_MM = []
            new_r_ball_MM = None

            if date_int in available_dates_MM:
                new_w_balls_MM = machine_MM[date_int]["W"]
                new_r_ball_MM = machine_MM[date_int]["R"]

            combined_w_balls = list(set(new_w_balls_PM + new_w_balls_MM))

            new_m = random.choice([2, 3, 4, 5, 10])

            # Update how 'r' is selected to ensure it's an integer
            combined_r_ball = None
            if isinstance(new_r_ball_PM, int) and isinstance(new_r_ball_MM, int):
                combined_r_ball = random.choice([new_r_ball_PM, new_r_ball_MM])
            elif isinstance(new_r_ball_PM, int):
                combined_r_ball = new_r_ball_PM
            elif isinstance(new_r_ball_MM, int):
                combined_r_ball = new_r_ball_MM
            else:
                # Both new_r_ball_PM and new_r_ball_MM are None, select random 'r' from respective ball ranges
                combined_r_ball = random.choice([random.choice(MM_R_BALLS), random.choice(PB_R_BALLS)])

            if len(combined_w_balls) >= 5:
                combined_w_balls = random.sample(combined_w_balls, 5)
            else:
                remaining_w_balls = list(set(MM_W_BALLS) - set(combined_w_balls))
                additional_w_balls_needed = 5 - len(combined_w_balls)
                combined_w_balls += random.sample(remaining_w_balls, additional_w_balls_needed)

            new_entries.append([date_int, combined_w_balls, combined_r_ball, new_m])

    return pd.DataFrame(new_entries, columns=["d", "w", "r", "m"])


# Fill in the missing entries in MM dataset
new_entries_mm = fill_missing_entries(mm, MM_PM, MM_MM)
mm = pd.concat([mm, new_entries_mm], ignore_index=True)

# Fill in the missing entries in PB dataset
new_entries_pb = fill_missing_entries(pb, PB_PM, PB_MM)
pb = pd.concat([pb, new_entries_pb], ignore_index=True)

# Output the updated datasets
print("Updated MM Dataset:")
print(mm.head())

print("Updated PB Dataset:")
print(pb.head())

# Calculate and print the elapsed time
end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time:.4f} seconds")
