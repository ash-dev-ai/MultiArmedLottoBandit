# draw1.py
import pandas as pd
import os
import matplotlib.pyplot as plt

class LotteryVisualizer:
    def __init__(self, file_path, max_num, max_numA, is_pb=False, is_comb=False):
        """
        Initialize the LotteryVisualizer with dataset parameters.
        
        :param file_path: Path to the CSV file.
        :param max_num: Maximum value for num1-num5.
        :param max_numA: Maximum value for numA.
        :param is_pb: Boolean flag for Powerball (True if Powerball, False if Mega Millions).
        :param is_comb: Boolean flag for combined data (if True, handles accordingly).
        """
        self.data = pd.read_csv(file_path)
        self.max_num = max_num
        self.max_numA = max_numA
        self.is_pb = is_pb
        self.is_comb = is_comb

    def plot_numbers(self, output_dir="images"):
        """
        Create line graphs for num1 to num5 and numA, both individually and combined.
        
        :param output_dir: Directory to save the images.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        columns_to_plot = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA']
        for column in columns_to_plot:
            plt.figure(figsize=(100, 30))
            plt.plot(self.data.index, self.data[column], label=column)
            plt.title(f"Line Graph of {column}")
            plt.xlabel("Index")
            plt.ylabel(column)
            plt.legend()
            plt.grid()
            output_path = os.path.join(output_dir, f"{column}_line_graph.png")
            plt.savefig(output_path)
            plt.close()
        
        # Plot all on one graph
        plt.figure(figsize=(140, 30))
        for column in columns_to_plot:
            plt.plot(self.data.index, self.data[column], label=column)
        plt.title("Line Graph of num1 to num5 and numA")
        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.legend()
        plt.grid()
        output_path = os.path.join(output_dir, "combined_line_graph.png")
        plt.savefig(output_path)
        plt.close()

