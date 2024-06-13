# analysis_main.py

import logging
import os
from analysis.proximity_analysis import ProximityAnalysis
from analysis.pattern_analysis import PatternAnalysis
from analysis.reward_penalty_system import RewardPenaltySystem
from analysis.visualization import Visualization

def ensure_directories_exist(directories):
    """Ensure that all specified directories exist. If not, create them."""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created missing directory: {directory}")

def run_proximity_analysis():
    """Run the proximity analysis."""
    output_dir = 'analysis/analysis_results/proximity'
    ensure_directories_exist([output_dir])
    
    proximity_analysis = ProximityAnalysis(
        predictions_dir='data/predictions',
        data_combined='data/data_combined.csv',
        data_pb='data/data_pb.csv',
        data_mb='data/data_mb.csv',
        output_dir=output_dir
    )
    proximity_analysis.run_analysis()

def run_pattern_analysis():
    """Run the pattern analysis."""
    analysis_dir = 'analysis/analysis_results/pattern'
    ensure_directories_exist([analysis_dir])
    
    pattern_analysis = PatternAnalysis(
        predictions_dir='data/predictions',
        data_dir='data',
        analysis_dir=analysis_dir
    )
    pattern_analysis.run_pattern_analysis()

def run_reward_penalty_system():
    """Run the reward and penalty system analysis."""
    analysis_dir = 'analysis/analysis_results/reward_penalty'
    ensure_directories_exist([analysis_dir])
    
    reward_penalty_system = RewardPenaltySystem(
        predictions_dir='data/predictions',
        data_dir='data',
        analysis_dir=analysis_dir
    )
    reward_penalty_system.run_reward_penalty_system()

def run_visualization():
    """Run the visualization process."""
    analysis_dir = 'analysis/analysis_results'
    output_dir = 'analysis/analysis_results/visualizations'
    ensure_directories_exist([output_dir])
    
    visualization = Visualization(
        predictions_dir='data/predictions',
        data_dir='data',
        analysis_dir=analysis_dir,
        output_dir=output_dir
    )
    visualization.run_visualization()

def main():
    """Main function to run all analyses and visualizations."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting the analysis process...")

    # Run the analysis processes
    try:
        logging.info("Starting proximity analysis...")
        run_proximity_analysis()
    except Exception as e:
        logging.error(f"Error in proximity analysis: {e}")

    try:
        logging.info("Starting pattern analysis...")
        run_pattern_analysis()
    except Exception as e:
        logging.error(f"Error in pattern analysis: {e}")

    try:
        logging.info("Starting reward and penalty system analysis...")
        run_reward_penalty_system()
    except Exception as e:
        logging.error(f"Error in reward and penalty system analysis: {e}")

    try:
        logging.info("Starting visualization process...")
        run_visualization()
    except Exception as e:
        logging.error(f"Error in visualization process: {e}")

    logging.info("Analysis process completed.")

if __name__ == "__main__":
    main()
