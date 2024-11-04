import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import colormaps
from datetime import datetime

class LotteryVisualizer:
    def __init__(self, file_path, max_num, max_numA, is_pb=False, is_comb=False):
        """
        Initialize the LotteryVisualizer with dataset parameters.
        
        :param file_path: Path to the CSV file.
        :param max_num: Maximum value for num1-num5.
        :param max_numA: Maximum value for numA.
        :param is_pb: Boolean flag for Powerball (True if Powerball, False if Mega Millions).
        """
        self.data = pd.read_csv(file_path)
        self.max_num = max_num
        self.max_numA = max_numA
        self.is_pb = is_pb
        self.is_comb = is_comb
        self.height = len(self.data)  # Number of rows in the dataset to determine the height of the image.
        self.date_str = datetime.now().strftime("%Y-%m-%d")  # Get current date as string
    
    def create_cumulative_image(self, output_dir="images", filename="cumulative_image.png", limit=None):
        """
        Create a cumulative image where each draw is represented as a line in the image.
        
        :param output_dir: Directory to save the image.
        :param filename: Name of the output image file.
        :param limit: Limit the number of rows processed. If None, all rows will be processed.
        """
        # Construct full output path
        output_dir = os.path.join(output_dir, "cumulative")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Add date to filename
        filename = f"{filename}_{self.date_str}.png"
        num_rows = len(self.data) if limit is None else min(limit, len(self.data))
        fig, ax = plt.subplots(figsize=(40, num_rows / 5))  # Adjusted width for better proportions
        ax.set_xlim(0, 8)  # Adjust x-limits for more balance
        ax.set_ylim(0, num_rows)

        # Color maps for continuous values (e.g., mean, pairwise comparisons)
        mean_cmap = colormaps.get_cmap('Greens')
        pairwise_cmap = colormaps.get_cmap('Blues')

        # Normalize values for gradient coloring
        mean_min, mean_max = self.data['mean'].min(), self.data['mean'].max()
        pairwise_min, pairwise_max = self.data[['N2-3', 'N3-5', 'N4-5']].min().min(), self.data[['N2-3', 'N3-5', 'N4-5']].max().max()

        # Plot each row as a line, one by one
        for index, row in self.data.iterrows():
            if index >= num_rows:
                break
            
            # Plot black blocks for selected num1-num5 in positions 0-5
            for num in range(1, 6):
                if row[f'num{num}'] <= self.max_num:
                    ax.add_patch(plt.Rectangle((num - 1, num_rows - index - 1), 1, 1, color='black'))
            
            # Plot red block for numA in position 6
            if row['numA'] <= self.max_numA:
                ax.add_patch(plt.Rectangle((5, num_rows - index - 1), 1, 1, color='red'))

            # Plot gradient blocks for 'mean' in position 7
            mean_color = mean_cmap((row['mean'] - mean_min) / (mean_max - mean_min))
            ax.add_patch(plt.Rectangle((6, num_rows - index - 1), 1, 1, color=mean_color))

            # Plot gradient blocks for pairwise comparisons in positions 8-10
            pairwise_columns = ['N2-3', 'N3-5', 'N4-5']
            for idx, col in enumerate(pairwise_columns):
                pairwise_value = (row[col] - pairwise_min) / (pairwise_max - pairwise_min)
                pairwise_color = pairwise_cmap(pairwise_value)
                ax.add_patch(plt.Rectangle((7 + idx, num_rows - index - 1), 1, 1, color=pairwise_color))

        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Save cumulative image
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    def apply_rule(self, rule_number, binary_string):
        """
        Apply a cellular automaton rule to a binary string to generate the next generation.
        
        :param rule_number: The rule number (e.g., 150, 110).
        :param binary_string: The current binary string representing the row.
        :return: A new binary string representing the next generation.
        """
        # Convert rule number to 8-bit binary representation
        rule_binary = f"{rule_number:08b}"
        rule_map = {f"{i:03b}": int(rule_binary[7 - i]) for i in range(8)}
        
        next_gen = ""
        for i in range(len(binary_string)):
            # Neighborhood (wrap around at edges)
            neighborhood = (
                binary_string[i - 1] + binary_string[i] + binary_string[(i + 1) % len(binary_string)]
            )
            next_gen += str(rule_map[neighborhood])
        
        return next_gen

    def create_automata_image(self, rule_number=150, output_dir="images", filename="automata_image.png", limit=100):
        """
        Create an automata-based image where each draw evolves according to a specified rule.
        
        :param rule_number: The cellular automaton rule number (e.g., 150).
        :param output_dir: Directory to save the image.
        :param filename: Name of the output image file.
        :param limit: Limit the number of rows processed. If None, all rows will be processed.
        """
        # Construct full output path
        output_dir = os.path.join(output_dir, "automata")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Add date to filename
        filename = f"{filename}_{self.date_str}.png"

        # Initialize first row based on the first draw (binary representation of num1-num5 presence)
        num_rows = len(self.data) if limit is None else min(limit, len(self.data))
        initial_state = ''.join(['1' if self.data.loc[0, f'num{n}'] <= self.max_num else '0' for n in range(1, 6)])
        generations = [initial_state]

        # Generate new rows based on cellular automaton rules
        for _ in range(num_rows - 1):
            next_gen = self.apply_rule(rule_number, generations[-1])
            generations.append(next_gen)

        fig, ax = plt.subplots(figsize=(20, num_rows / 10))  # Adjust figure size for readability

        # Visualize each generation as a line of patches
        for idx, generation in enumerate(generations):
            for jdx, cell in enumerate(generation):
                color = 'black' if cell == '1' else 'white'
                ax.add_patch(plt.Rectangle((jdx, num_rows - idx - 1), 1, 1, color=color))

        # Remove axes for clean visualization
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Save the automata image
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

# Example usage for Mega Millions with Rule 150
mb_visualizer = LotteryVisualizer(file_path="D:\Projects\Project-PBMM\current\MultiArmedLottoBandit\data\data_mb.csv", max_num=70, max_numA=25)
mb_visualizer.create_cumulative_image(output_dir="images/mb_images", filename="mb_cumulative_image", limit=600)
mb_visualizer.create_automata_image(rule_number=150, output_dir="images/mb_automata", filename="mb_automata_image", limit=100)

# Example usage for Powerball with Rule 110
pb_visualizer = LotteryVisualizer(file_path="D:\Projects\Project-PBMM\current\MultiArmedLottoBandit\data\data_pb.csv", max_num=69, max_numA=26, is_pb=True)
pb_visualizer.create_cumulative_image(output_dir="images/pb_images", filename="pb_cumulative_image", limit=600)
pb_visualizer.create_automata_image(rule_number=110, output_dir="images/pb_automata", filename="pb_automata_image", limit=100)

# Example usage for Combined dataset
comb_visualizer = LotteryVisualizer(file_path="D:\Projects\Project-PBMM\current\MultiArmedLottoBandit\data\data_combined.csv", max_num=70, max_numA=26, is_comb=True)
comb_visualizer.create_cumulative_image(output_dir="images/comb_images", filename="comb_cumulative_image", limit=700)
comb_visualizer.create_automata_image(rule_number=150, output_dir="images/comb_automata", filename="comb_automata_image", limit=100)
