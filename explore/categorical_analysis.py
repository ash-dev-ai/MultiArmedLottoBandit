# categorical_analysis.py
import matplotlib.pyplot as plt
import seaborn as sns
import os

def categorical_analysis(data, name):
    """Analyze and visualize categorical variables."""
    plot_dir = f'explore/plots/{name}'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Frequency counts for weekdays
    plt.figure(figsize=(50, 10))
    sns.countplot(x='weekday', data=data)
    plt.savefig(f'{plot_dir}/weekday_counts.png')

# No main function, this script will be called from explore_main.py

