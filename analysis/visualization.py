# visualization.py

import logging
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:
    def __init__(self, predictions_dir, data_dir, analysis_dir, output_dir):
        """
        Initialize the Visualization class with directories for predictions, data, analysis results, and where to save plots.

        Parameters:
        - predictions_dir: Directory containing prediction CSV files.
        - data_dir: Directory containing actual data CSV files.
        - analysis_dir: Directory to save the analysis results.
        - output_dir: Directory to save the generated plots.
        """
        self.predictions_dir = predictions_dir
        self.data_dir = data_dir
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run_visualization(self):
        """Run all visualization processes."""
        logging.info("Starting visualization process...")
        
        # Proximity Analysis Visualization
        self.proximity_visualization()
        
        # Pattern Analysis Visualization
        self.pattern_visualization()
        
        # Reward/Penalty Visualization
        self.reward_penalty_visualization()
        
        logging.info("Visualization process completed.")

    def proximity_visualization(self):
        """Visualize the proximity analysis results."""
        logging.info("Visualizing proximity analysis...")

        # Find the latest proximity file in the 'proximity' subdirectory
        proximity_dir = os.path.join(self.analysis_dir, 'proximity')
        proximity_file = self.get_latest_file(proximity_dir, 'proximity_analysis_results_')

        if not proximity_file:
            logging.error(f"No proximity analysis results found in {proximity_dir}.")
            return

        try:
            proximity_scores = pd.read_csv(proximity_file)
            if 'dataset_type' not in proximity_scores.columns or 'proximity_score' not in proximity_scores.columns:
                raise ValueError("Required columns missing in proximity analysis file.")

            sns.boxplot(x='dataset_type', y='proximity_score', data=proximity_scores)
            plt.title('Proximity Scores Across Datasets')
            plt.xlabel('Dataset Type')
            plt.ylabel('Proximity Score')
            plt.xticks(rotation=45)
            self.save_plot('proximity_scores')
        except Exception as e:
            logging.error(f"Error processing proximity analysis file {proximity_file}: {e}")

    def pattern_visualization(self):
        """Visualize the pattern analysis results."""
        logging.info("Visualizing pattern analysis...")

        # Find all pattern analysis summary files in the 'pattern' subdirectory
        pattern_dir = os.path.join(self.analysis_dir, 'pattern')
        pattern_files = self.get_files_with_suffix(pattern_dir, '_pattern_summary.csv')

        if not pattern_files:
            logging.error(f"No pattern analysis summary files found in {pattern_dir}.")
            return

        try:
            summaries = []
            for file in pattern_files:
                summary = pd.read_csv(file)
                model_dataset = os.path.basename(file).replace('_pattern_summary.csv', '')
                summary['model_dataset'] = model_dataset
                summaries.append(summary)

            combined_summary = pd.concat(summaries, axis=0)
            sns.barplot(x='Number', y='Frequency', hue='model_dataset', data=combined_summary)
            plt.title('Common Patterns Detected in Predictions')
            plt.xlabel('Number')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45)
            self.save_plot('pattern_analysis')
        except Exception as e:
            logging.error(f"Error processing pattern analysis files: {e}")

    def reward_penalty_visualization(self):
        """Visualize the reward and penalty system results."""
        logging.info("Visualizing reward and penalty system...")

        # Find all reward_penalty summary files in the 'reward_penalty' subdirectory
        reward_penalty_dir = os.path.join(self.analysis_dir, 'reward_penalty')
        reward_penalty_files = self.get_files_with_suffix(reward_penalty_dir, '_reward_penalty_summary.csv')

        if not reward_penalty_files:
            logging.error(f"No reward penalty summary files found in {reward_penalty_dir}.")
            return

        try:
            summaries = []
            for file in reward_penalty_files:
                summary = pd.read_csv(file)
                model_dataset = os.path.basename(file).replace('_reward_penalty_summary.csv', '')
                summary['model_dataset'] = model_dataset
                summaries.append(summary)

            combined_summary = pd.concat(summaries, axis=0)
            sns.barplot(x='model_dataset', y='total_points', data=combined_summary)
            plt.title('Total Points by Model and Dataset')
            plt.xlabel('Model and Dataset')
            plt.ylabel('Total Points')
            plt.xticks(rotation=45)
            self.save_plot('reward_penalty_system')
        except Exception as e:
            logging.error(f"Error processing reward and penalty files: {e}")

    def save_plot(self, plot_name):
        """Save the current plot to a file."""
        plot_path = os.path.join(self.output_dir, f'{plot_name}.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.clf()  # Clear the current figure to avoid overlap in subsequent plots
        logging.info(f"Plot saved to {plot_path}")

    def get_latest_file(self, directory, prefix):
        """Get the most recent file in the specified directory that starts with the given prefix."""
        files = [f for f in os.listdir(directory) if f.startswith(prefix)]
        if not files:
            return None
        files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
        return os.path.join(directory, files[0])

    def get_files_with_suffix(self, directory, suffix):
        """Get all files in the specified directory that end with the given suffix."""
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(suffix)]

# Example of how this class could be instantiated and run:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    predictions_dir = 'data/predictions'
    data_dir = 'data'
    analysis_dir = 'analysis/analysis_results'
    output_dir = 'analysis/visualizations'

    visualization = Visualization(predictions_dir, data_dir, analysis_dir, output_dir)
    visualization.run_visualization()

