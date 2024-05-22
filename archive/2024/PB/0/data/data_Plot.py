import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_interIntra import number_tracker, org_pb

# Time Series Plot for 'w'
def plot_time_series_w():
    # Create an empty DataFrame to store frequency data for white balls
    time_series_df_w = pd.DataFrame(index=org_pb['d'].unique())
    
    # Calculate frequency for each white ball number and add it to the DataFrame
    for num, dates in number_tracker.items():
        time_series_df_w[num] = time_series_df_w.index.to_series().apply(lambda x: dates.count(x))
    
    # Plotting individual Time Series for each number in 'w'
    for num in range(1, 71):
        plt.figure(figsize=(10, 5))
        plt.plot(time_series_df_w.index, time_series_df_w[num])
        plt.title(f"Time Series for White Ball Number {num}")
        plt.xlabel("Date")
        plt.ylabel("Frequency")
        plt.show()

# Time Series Plot for 'r'
def plot_time_series_r():
    # Plotting Time Series for 'r' tracing its actual values over time
    plt.figure(figsize=(10, 5))
    plt.plot(org_pb['d'], org_pb['r'])
    plt.title("Time Series for Red Ball")
    plt.xlabel("Date")
    plt.ylabel("Red Ball Number")
    plt.show()

# Heatmap for 'w'
def plot_heatmap_w():
    # Create a DataFrame similar to time_series_df but with columns being dates
    heatmap_df_w = pd.DataFrame(index=range(1, 71))
    for date in org_pb['d'].unique():
        heatmap_df_w[date] = heatmap_df_w.index.to_series().apply(lambda x: number_tracker[x].count(date))
    
    # Plotting
    plt.figure(figsize=(20, 10))
    sns.heatmap(heatmap_df_w, cmap='viridis', cbar=True)
    plt.title("Heatmap of White Ball Number Frequencies")
    plt.xlabel("Date")
    plt.ylabel("Number")
    plt.show()

# Heatmap for 'r'
def plot_heatmap_r():
    # Create a DataFrame to store frequency data for red ball
    heatmap_df_r = pd.DataFrame(index=range(1, 27))
    for date in org_pb['d'].unique():
        heatmap_df_r[date] = heatmap_df_r.index.to_series().apply(lambda x: (org_pb[org_pb['d'] == date]['r'] == x).sum())
    
    # Plotting
    plt.figure(figsize=(20, 10))
    sns.heatmap(heatmap_df_r, cmap='viridis', cbar=True)
    plt.title("Heatmap of Red Ball Frequencies")
    plt.xlabel("Date")
    plt.ylabel("Red Ball Number")
    plt.show()

# Histogram for 'w'
def plot_histogram_w():
    # Create an empty list to store all inter-draw differences for 'w'
    all_inter_draw_diffs_w = []
    for i in range(1, len(org_pb['w'])):
        curr_draw = org_pb['w'].iloc[i]
        prev_draw = org_pb['w'].iloc[i-1]
        all_inter_draw_diffs_w.extend([c - p for c, p in zip(curr_draw, prev_draw)])
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.hist(all_inter_draw_diffs_w, bins=20, edgecolor='black')
    plt.title("Histogram of Inter-Draw Differences for White Ball Numbers")
    plt.xlabel("Inter-Draw Differences")
    plt.ylabel("Frequency")
    plt.show()

# Execute the plotting functions
plot_time_series_w()
plot_time_series_r()
plot_heatmap_w()
plot_heatmap_r()
plot_histogram_w()