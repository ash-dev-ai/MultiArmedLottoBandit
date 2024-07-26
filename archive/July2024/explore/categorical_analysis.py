# categorical_analysis.py

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

class CategoricalAnalysis:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.plot_dir = f'explore/plots/{name}'
        self.summary_dir = f'explore/summaries/{name}'
        self.ensure_dir(self.plot_dir)
        self.ensure_dir(self.summary_dir)

    def ensure_dir(self, directory):
        """Ensure that a directory exists."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def plot_categorical(self, column):
        """Plot and save a categorical count plot."""
        plt.figure(figsize=(10, 10))
        sns.countplot(x=column, data=self.data)
        plt.title(f'{column.capitalize()} Counts')
        plt.savefig(f'{self.plot_dir}/{column}_counts.png')
        plt.close()

    def save_categorical_summary(self, column):
        """Save summary statistics for a categorical column."""
        summary = self.data[column].value_counts().reset_index()
        summary.columns = [column, 'count']
        summary.to_csv(f'{self.summary_dir}/{column}_summary.csv', index=False)

    def analyze(self):
        """Analyze and visualize categorical variables."""
        categorical_columns = ['weekday']  # Add other categorical columns as needed
        for column in categorical_columns:
            self.plot_categorical(column)
            self.save_categorical_summary(column)

# No main function, this script will be called from explore_main.py
