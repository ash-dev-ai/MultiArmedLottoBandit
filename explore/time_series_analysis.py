# time_series_analysis.py
# time_series_analysis.py
import matplotlib.pyplot as plt
import os
import pandas as pd
import logging

class TimeSeriesAnalysis:
    def __init__(self, data: pd.DataFrame, name: str):
        self.data = data
        self.name = name
        self.plot_dir = f'explore/plots/{name}'
        os.makedirs(self.plot_dir, exist_ok=True)
        self.data['draw_date'] = pd.to_datetime(self.data['draw_date'])
        self.data.set_index('draw_date', inplace=True)
    
    def plot_numSum_over_time(self):
        """Plot and save numSum over time."""
        plt.figure(figsize=(20, 20))
        self.data['numSum'].plot()
        plt.title('numSum Over Time')
        plt.savefig(f'{self.plot_dir}/numSum_over_time.png')
        plt.close()
        logging.info(f"numSum over time plot saved at {self.plot_dir}/numSum_over_time.png")
    
    def plot_totalSum_over_time(self):
        """Plot and save totalSum over time."""
        plt.figure(figsize=(20, 20))
        self.data['totalSum'].plot()
        plt.title('totalSum Over Time')
        plt.savefig(f'{self.plot_dir}/totalSum_over_time.png')
        plt.close()
        logging.info(f"totalSum over time plot saved at {self.plot_dir}/totalSum_over_time.png")
        
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics of the time series."""
        summary_stats = self.data[['numSum', 'totalSum']].describe()
        logging.info(f"Summary statistics for {self.name} dataset:\n{summary_stats}")
        return summary_stats
    
    def save_summary_statistics(self):
        """Save summary statistics to a CSV file."""
        summary_stats = self.get_summary_statistics()
        summary_stats.to_csv(f'{self.plot_dir}/summary_statistics.csv')
        logging.info(f"Summary statistics saved at {self.plot_dir}/summary_statistics.csv")
        
    def analyze(self):
        """Run all analysis methods."""
        self.plot_numSum_over_time()
        self.plot_totalSum_over_time()
        self.save_summary_statistics()

# No main function, this script will be called from explore_main.py



