# time_series_analysis.py
import matplotlib.pyplot as plt
import os
import pandas as pd

def time_series_analysis(data, name):
    """Analyze and visualize time series data."""
    plot_dir = f'explore/plots/{name}'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Line plot of numSum over time
    data['draw_date'] = pd.to_datetime(data['draw_date'])
    data.set_index('draw_date', inplace=True)
    
    plt.figure(figsize=(50, 50))
    data['numSum'].plot()
    plt.savefig(f'{plot_dir}/numSum_over_time.png')
    
    # Line plot of totalSum over time
    plt.figure(figsize=(50, 50))
    data['totalSum'].plot()
    plt.savefig(f'{plot_dir}/totalSum_over_time.png')

# No main function, this script will be called from explore_main.py



