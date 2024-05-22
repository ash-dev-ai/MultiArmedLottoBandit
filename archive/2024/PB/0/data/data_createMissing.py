import random
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from dateutil import parser

# Record the start time
start_time = time.process_time()

# The import for datasets
from data_import import data_pb

pb = pd.DataFrame(data_pb, columns=["d", "w", "r", "m"])
pb["r"] = pb["r"].astype(int)

# Convert datetime to an integer representing days since the earliest entry in the d column
# Assuming the earliest entry is the minimum date in both datasets
earliest_date_pb = pb["d"].min()

def datetime_to_days_since_earliest(date, earliest_date):
    return (date - earliest_date).days + 1

pb["d"] = pb["d"].apply(lambda x: datetime_to_days_since_earliest(x, earliest_date_pb))

# PB Configuration
earliest_date_pb_int = datetime_to_days_since_earliest(earliest_date_pb, earliest_date_pb)
PB_W_BALLS = list(range(1, 70))  # White balls labeled 1-69
PB_R_BALLS = list(range(1, 27))  # Red balls labeled 1-26

# Function to create the gravity pick machine
def create_gravity_pick_machine(earliest_date_int, W_BALLS, R_BALLS):
    # Ensure R_BALLS only contains integers
    R_BALLS = [int(ball) for ball in R_BALLS]
    
    machine_PM = {}
    for date_int in range(earliest_date_int, earliest_date_int + 1000):
        W_machine_balls_PM = random.sample(W_BALLS, 5)
        R_machine_ball_PM = random.choice(R_BALLS)
        machine_PM[date_int] = {"W": W_machine_balls_PM, "R": R_machine_ball_PM}
    return machine_PM

# Create PB gravity pick machines
PB_PM = create_gravity_pick_machine(earliest_date_pb_int, PB_W_BALLS, PB_R_BALLS)

# Get the current date and convert it to days since the earliest date
current_date = datetime.now()
earliest_date_pb_datetime = parser.parse(str(earliest_date_pb))

current_date_int = int(datetime_to_days_since_earliest(current_date, earliest_date_pb_datetime))  # Cast to int

# Function to fill in the missing entries in the datasets using the gravity pick machines
def fill_missing_entries(df, machine_PM, current_date_int):
    new_entries = []
    all_dates = list(range(df["d"].min(), current_date_int + 1))  # Fix the date range calculation
    available_dates_PM = list(machine_PM.keys())
    for date_int in all_dates:
        if date_int not in df["d"].values:
            new_w_balls_PM = []
            new_r_ball_PM = None

            if date_int in available_dates_PM:
                new_w_balls_PM = machine_PM[date_int]["W"]
                new_r_ball_PM = machine_PM[date_int]["R"]

            combined_w_balls = list(set(new_w_balls_PM))

            new_m = random.choice([2, 3, 4, 5, 10])

            combined_r_ball = None
            if isinstance(new_r_ball_PM, int):
                combined_r_ball = new_r_ball_PM
            else:
                combined_r_ball = random.choice(PB_R_BALLS)

            if len(combined_w_balls) >= 5:
                combined_w_balls = random.sample(combined_w_balls, 5)
            else:
                remaining_w_balls = list(set(PB_W_BALLS) - set(combined_w_balls))
                additional_w_balls_needed = 5 - len(combined_w_balls)
                combined_w_balls += random.sample(remaining_w_balls, additional_w_balls_needed)

            new_entries.append([date_int, combined_w_balls, combined_r_ball, new_m])

    return pd.DataFrame(new_entries, columns=["d", "w", "r", "m"])

# Fill in the missing entries in PB dataset
new_entries_pb = fill_missing_entries(pb, PB_PM, current_date_int)  # Pass current_date_int as an argument
pb = pd.concat([pb, new_entries_pb], ignore_index=True)

print("Updated PB Dataset:")
print(pb.head())

# Calculate and print the elapsed time
end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time:.4f} seconds")