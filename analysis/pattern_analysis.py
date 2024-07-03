# pattern_analysis.py

import logging
import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt

class PatternAnalysis:
    def __init__(self, predictions_dir, data_dir, analysis_dir):
        """
        Initialize the PatternAnalysis with directories for predictions, data, and where to save analysis results.

        Parameters:
        - predictions_dir: Directory containing prediction CSV files.
        - data_dir: Directory containing actual data CSV files.
        - analysis_dir: Directory to save the analysis results.
        """
        self.predictions_dir = predictions_dir
        self.data_dir = data_dir
        self.analysis_dir = analysis_dir

        # Ensure the analysis directory exists
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)

    def run_pattern_analysis(self):
        logging.info("Running pattern analysis...")

        # Load predictions
        predictions = self.load_predictions()

        # Load actuals
        actuals_combined = pd.read_csv(os.path.join(self.data_dir, 'data_combined.csv'))
        actuals_pb = pd.read_csv(os.path.join(self.data_dir, 'data_pb.csv'))
        actuals_mb = pd.read_csv(os.path.join(self.data_dir, 'data_mb.csv'))

        # Analyze and compare patterns
        self.analyze_and_compare(predictions, actuals_combined, 'combined')
        self.analyze_and_compare(predictions, actuals_pb, 'pb')
        self.analyze_and_compare(predictions, actuals_mb, 'mb')

        logging.info("Pattern analysis completed.")

    def load_predictions(self):
        """
        Load prediction files and organize them by dataset type.

        Returns:
        A dictionary with dataset types ('combined', 'pb', 'mb') as keys and model prediction DataFrames as values.
        """
        predictions = {'combined': {}, 'pb': {}, 'mb': {}}
        files = [f for f in os.listdir(self.predictions_dir) if f.endswith('.csv')]
        
        if not files:
            logging.error(f"No prediction files found in {self.predictions_dir}.")
            return predictions

        for file in files:
            path = os.path.join(self.predictions_dir, file)
            model_name, dataset_type = file.split('_')[:2]
            if dataset_type in predictions:
                predictions[dataset_type][model_name] = pd.read_csv(path)

        return predictions

    def analyze_and_compare(self, predictions, actuals, dataset_type):
        """
        Analyze patterns in predictions and compare with actual data.

        Parameters:
        - predictions: Dictionary of predictions organized by model.
        - actuals: DataFrame of actual data.
        - dataset_type: Type of dataset being analyzed ('combined', 'pb', 'mb').
        """
        logging.info(f"Analyzing and comparing patterns for {dataset_type} dataset...")

        # Analyze patterns in predictions
        for model, preds in predictions[dataset_type].items():
            pattern_summary = self.analyze_patterns(preds, model, dataset_type)
            # Compare with actual data
            self.compare_with_actuals(pattern_summary, actuals, model, dataset_type)

    def analyze_patterns(self, preds, model, dataset_type):
        """
        Analyze number patterns in predictions.

        Parameters:
        - preds: DataFrame of predictions.
        - model: Name of the prediction model.
        - dataset_type: Type of dataset being analyzed.

        Returns:
        A dictionary summarizing the most common, least common numbers, mean, and median.
        """
        logging.info(f"Analyzing patterns for {model} model ({dataset_type} dataset)...")

        # Determine the relevant columns based on the presence of 'num1'
        if 'num1' in preds.columns:
            all_numbers = preds[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].values.flatten()
        else:
            all_numbers = pd.concat([preds['numSum'], preds['totalSum']]).values.flatten()

        # Calculate frequency of each number
        number_counts = Counter(all_numbers)

        # Statistical summaries
        most_common_numbers = number_counts.most_common(5)  # Top 5 most common numbers
        least_common_numbers = number_counts.most_common()[-5:]  # Least 5 common numbers
        mean_value = pd.Series(all_numbers).mean()
        median_value = pd.Series(all_numbers).median()

        pattern_summary = {
            'most_common': most_common_numbers,
            'least_common': least_common_numbers,
            'mean': mean_value,
            'median': median_value
        }

        # Save patterns to CSV
        self.save_pattern_summary(model, dataset_type, pattern_summary)

        # Plot frequency distribution
        self.plot_number_distribution(number_counts, model, dataset_type)

        return pattern_summary

    def compare_with_actuals(self, pattern_summary, actuals, model, dataset_type):
        """
        Compare prediction patterns with actual data.

        Parameters:
        - pattern_summary: Dictionary summarizing the prediction patterns.
        - actuals: DataFrame of actual data.
        - model: Name of the prediction model.
        - dataset_type: Type of dataset being analyzed.
        """
        logging.info(f"Comparing prediction patterns with actual results for {dataset_type} dataset...")

        # Extract actual numbers
        actual_numbers = actuals[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].values.flatten()
        actual_counts = Counter(actual_numbers)

        logging.info(f"Model: {model}")
        logging.info(f"Most Common Predicted Numbers: {pattern_summary['most_common']}")
        logging.info(f"Most Common Actual Numbers: {actual_counts.most_common(5)}")
        logging.info(f"Mean Predicted Value: {pattern_summary['mean']:.2f}")
        logging.info(f"Mean Actual Value: {pd.Series(actual_numbers).mean():.2f}")
        logging.info(f"Median Predicted Value: {pattern_summary['median']:.2f}")
        logging.info(f"Median Actual Value: {pd.Series(actual_numbers).median():.2f}")

    def save_pattern_summary(self, model, dataset_type, summary):
        """
        Save the pattern summary to a CSV file.

        Parameters:
        - model: Name of the prediction model.
        - dataset_type: Type of dataset being analyzed.
        - summary: Dictionary summarizing the prediction patterns.
        """
        summary_file = os.path.join(self.analysis_dir, f'{model}_{dataset_type}_pattern_summary.csv')
        summary_df = pd.DataFrame(summary['most_common'], columns=['Number', 'Frequency'])
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"Pattern summary for {model} ({dataset_type}) saved to {summary_file}")

    def plot_number_distribution(self, number_counts, model, dataset_type):
        """
        Plot and save the number frequency distribution.

        Parameters:
        - number_counts: Counter of number frequencies.
        - model: Name of the prediction model.
        - dataset_type: Type of dataset being analyzed.
        """
        plt.figure(figsize=(10, 6))
        numbers, frequencies = zip(*number_counts.items())
        plt.bar(numbers, frequencies)
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.title(f'Number Frequency Distribution for {model} ({dataset_type})')
        plot_path = os.path.join(self.analysis_dir, f'{model}_{dataset_type}_number_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Number distribution plot for {model} ({dataset_type}) saved to {plot_path}")

# Example of how this class could be instantiated and run:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    pattern_analysis = PatternAnalysis(predictions_dir='data/predictions', data_dir='data', analysis_dir='analysis_results')
    pattern_analysis.run_pattern_analysis()
