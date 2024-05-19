# data_visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import os

def data_visualization(data, name):
    """Generate and save visualizations for the dataset."""
    plot_dir = f'explore/plots/{name}'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Histograms
    data.hist(figsize=(50, 50))
    plt.savefig(f'{plot_dir}/histograms.png')
    
    # Box plots
    data.boxplot(figsize=(50, 50))
    plt.savefig(f'{plot_dir}/boxplots.png')
    
    # Correlation heatmap
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(50, 50))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
    plt.savefig(f'{plot_dir}/correlation_heatmap.png')

# No main function, this script will be called from explore_main.py


