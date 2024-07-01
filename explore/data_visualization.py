# data_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

class DataVisualization:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.plot_dir = f'explore/plots/{name}'
        self.ensure_dir(self.plot_dir)

    def ensure_dir(self, directory):
        """Ensure that a directory exists."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def plot_histograms(self):
        """Generate and save histograms for the dataset."""
        self.data.hist(figsize=(20, 20))
        plt.savefig(f'{self.plot_dir}/histograms.png')
        plt.close()

    def plot_boxplots(self):
        """Generate and save box plots for the dataset."""
        self.data.boxplot(figsize=(20, 20))
        plt.savefig(f'{self.plot_dir}/boxplots.png')
        plt.close()

    def plot_correlation_heatmap(self):
        """Generate and save a correlation heatmap for the dataset."""
        numeric_data = self.data.select_dtypes(include=['int64', 'float64'])
        plt.figure(figsize=(20, 20))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
        plt.savefig(f'{self.plot_dir}/correlation_heatmap.png')
        plt.close()

    def plot_pairplot(self):
        """Generate and save a pair plot for the dataset."""
        sns.pairplot(self.data)
        plt.savefig(f'{self.plot_dir}/pairplot.png')
        plt.close()

    def plot_kde(self):
        """Generate and save KDE plots for the dataset."""
        plt.figure(figsize=(20, 20))
        for column in self.data.select_dtypes(include=['int64', 'float64']).columns:
            sns.kdeplot(self.data[column], shade=True)
        plt.savefig(f'{self.plot_dir}/kde_plots.png')
        plt.close()

    def plot_violin(self):
        """Generate and save violin plots for the dataset."""
        plt.figure(figsize=(20, 20))
        sns.violinplot(data=self.data.select_dtypes(include=['int64', 'float64']))
        plt.savefig(f'{self.plot_dir}/violin_plots.png')
        plt.close()

    def plot_line(self):
        """Generate and save line plots for the dataset."""
        plt.figure(figsize=(20, 20))
        for column in self.data.select_dtypes(include=['int64', 'float64']).columns:
            sns.lineplot(data=self.data, x=self.data.index, y=column)
        plt.savefig(f'{self.plot_dir}/line_plots.png')
        plt.close()

    def visualize(self):
        """Generate and save all visualizations."""
        self.plot_histograms()
        self.plot_boxplots()
        self.plot_correlation_heatmap()
        self.plot_pairplot()
        self.plot_kde()
        self.plot_violin()
        self.plot_line()

# No main function, this script will be called from explore_main.py
