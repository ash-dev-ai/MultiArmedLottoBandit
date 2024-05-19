# missing_values_analysis.py
import matplotlib.pyplot as plt
import seaborn as sns
import os

def missing_values_analysis(data, name):
    """Identify and visualize missing values."""
    plot_dir = f'explore/plots/{name}'
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(50, 50))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.savefig(f'{plot_dir}/missing_values.png')

# No main function, this script will be called from explore_main.py


