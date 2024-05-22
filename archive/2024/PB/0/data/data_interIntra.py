import pandas as pd
import numpy as np
from data_processWin import org_pb

def calculate_draw_differences(df):
    # Check the type of data in 'w' column
    first_entry = df['w'].iloc[0]
    if isinstance(first_entry, str):
        df['w'] = df['w'].apply(lambda x: sorted(list(map(int, x.split()))))
    elif isinstance(first_entry, list):
        df['w'] = df['w'].apply(lambda x: sorted(x))
    
    # Calculate differences between each pair of numbers within the same draw (intra-draw differences)
    df['Intra-Draw Differences'] = df['w'].apply(lambda x: [y - x[i - 1] for i, y in enumerate(x)][1:])
    
    # Calculate differences between the same positions in consecutive draws manually (inter-draw differences)
    inter_draw_diffs = []
    for i in range(1, len(df['w'])):
        curr_draw = df['w'][i]
        prev_draw = df['w'][i-1]
        inter_draw_diffs.append([c - p for c, p in zip(curr_draw, prev_draw)])
    df['Inter-Draw Differences'] = [np.nan] + inter_draw_diffs  # Add np.nan for the first entry
    
    # Calculate inter-draw differences for the red Powerball number ('r')
    df['Inter-Draw Red Differences'] = df['r'].diff()
    
    # Initialize a dictionary to track each possible number
    number_tracker = {i: [] for i in range(1, 71)}  # Assuming the possible numbers are from 1 to 70
    
    # Fill in the dates for each number
    for _, row in df.iterrows():
        draw_date = row['d']
        winning_numbers = row['w']
        for num in winning_numbers:
            number_tracker[num].append(draw_date)
            
    return df, number_tracker

# Execute the function and get the number_tracker
_, number_tracker = calculate_draw_differences(org_pb)