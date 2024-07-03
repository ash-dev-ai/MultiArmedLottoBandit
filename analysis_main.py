# analysis_main.py

import logging
import os
from analysis.proximity_analysis import ProximityAnalysis
from analysis.pattern_analysis import PatternAnalysis
from analysis.reward_penalty_system import RewardPenaltySystem
from analysis.visualization import Visualization

class AnalysisManager:
    def __init__(self):
        self.directories = {
            'proximity': 'analysis/analysis_results/proximity',
            'pattern': 'analysis/analysis_results/pattern',
            'reward_penalty': 'analysis/analysis_results/reward_penalty',
            'visualization': 'analysis/analysis_results/visualizations'
        }
        self.ensure_directories_exist()

    def ensure_directories_exist(self):
        """Ensure that all specified directories exist. If not, create them."""
        for directory in self.directories.values():
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Created missing directory: {directory}")

    def run_proximity_analysis(self):
        """Run the proximity analysis."""
        proximity_analysis = ProximityAnalysis(
            predictions_dir='data/predictions',
            data_combined='data/data_combined.csv',
            data_pb='data/data_pb.csv',
            data_mb='data/data_mb.csv',
            output_dir=self.directories['proximity']
        )
        proximity_analysis.run_analysis()

    def run_pattern_analysis(self):
        """Run the pattern analysis."""
        pattern_analysis = PatternAnalysis(
            predictions_dir='data/predictions',
            data_dir='data',
            analysis_dir=self.directories['pattern']
        )
        pattern_analysis.run_pattern_analysis()

    def run_reward_penalty_system(self):
        """Run the reward and penalty system analysis."""
        reward_penalty_system = RewardPenaltySystem(
            predictions_dir='data/predictions',
            data_dir='data',
            analysis_dir=self.directories['reward_penalty']
        )
        reward_penalty_system.run_reward_penalty_system()

    def run_visualization(self):
        """Run the visualization process."""
        visualization = Visualization(
            predictions_dir='data/predictions',
            data_dir='data',
            analysis_dir='analysis/analysis_results',
            output_dir=self.directories['visualization']
        )
        visualization.run_visualization()

    def run_all_analyses(self):
        """Run all analyses and visualizations."""
        try:
            logging.info("Starting proximity analysis...")
            self.run_proximity_analysis()
        except Exception as e:
            logging.error(f"Error in proximity analysis: {e}")

        try:
            logging.info("Starting pattern analysis...")
            self.run_pattern_analysis()
        except Exception as e:
            logging.error(f"Error in pattern analysis: {e}")

        try:
            logging.info("Starting reward and penalty system analysis...")
            self.run_reward_penalty_system()
        except Exception as e:
            logging.error(f"Error in reward and penalty system analysis: {e}")

        try:
            logging.info("Starting visualization process...")
            self.run_visualization()
        except Exception as e:
            logging.error(f"Error in visualization process: {e}")

        logging.info("Analysis process completed.")

def main():
    """Main function to run all analyses and visualizations."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the analysis process...")
    
    analysis_manager = AnalysisManager()
    analysis_manager.run_all_analyses()

if __name__ == "__main__":
    main()
