import pandas as pd
import matplotlib.pyplot as plt
import os

class LotteryVisualizer:
    def __init__(self, file_path, max_num, max_numA, is_pb=False):
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
        self.height = len(self.data)  # Number of rows in the dataset to determine the height of the image.
    
    def create_cumulative_image(self, output_dir=".", filename="cumulative_image.png", limit=None):
        """
        Create a cumulative image where each draw is represented as a line in the image.
        
        :param output_dir: Directory to save the image.
        :param filename: Name of the output image file.
        :param limit: Limit the number of rows processed. If None, all rows will be processed.
        """
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        num_rows = len(self.data) if limit is None else min(limit, len(self.data))
        fig, ax = plt.subplots(figsize=(70, num_rows / 5))  # Dynamic height based on number of rows
        ax.set_xlim(0, self.max_num)
        ax.set_ylim(0, num_rows)
        
        # Plot each row as a line, one by one
        for index, row in self.data.iterrows():
            if index >= num_rows:
                break
            
            # Plot black blocks for selected num1-num5
            for num in range(1, 6):
                if row[f'num{num}'] <= self.max_num:
                    ax.add_patch(plt.Rectangle((row[f'num{num}']-1, num_rows - index - 1), 1, 1, color='black'))
            
            # Plot red block for numA
            if row['numA'] <= self.max_numA:
                ax.add_patch(plt.Rectangle((row['numA']-1, num_rows - index - 1), 1, 1, color='red'))
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Save cumulative image
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        plt.close(fig)

# Example usage for Mega Millions
mb_visualizer = LotteryVisualizer(file_path="D:\Projects\Project-PBMM\current\MultiArmedLottoBandit\data\data_mb.csv", max_num=70, max_numA=25)
mb_visualizer.create_cumulative_image(output_dir="mb_images", filename="mb_cumulative_image.png", limit=600)

# Example usage for Powerball
pb_visualizer = LotteryVisualizer(file_path="D:\Projects\Project-PBMM\current\MultiArmedLottoBandit\data\data_pb.csv", max_num=69, max_numA=26, is_pb=True)
pb_visualizer.create_cumulative_image(output_dir="pb_images", filename="pb_cumulative_image.png", limit=600)
