import pandas as pd
import matplotlib.pyplot as plt
from data_processWin import org_pb

# Function to generate separate 3D plots, one for each w1, w2, ..., w5
def plot_3d_final_adjusted():
    for i in range(1, 6):  # Looping through w1, w2, ..., w5
        fig = plt.figure(figsize=(100, 150))
        ax = fig.add_subplot(311, projection='3d')
        
        # Using 'r' as x-axis, 'd' as y-axis, and each 'wi' on z-axis
        ax.scatter(org_pb['r'], org_pb['d'], org_pb[f'w{i}'], marker='1', s=300, depthshade=True)
        ax.set_xlabel('Red Ball')
        ax.set_ylabel('Date')
        ax.set_zlabel(f'White Ball {i}')
        ax.set_title(f'3D Plot for Red Ball, Date, and White Ball {i}')
        
        plt.show()

# Generate the separate 3D plots with final adjustments
plot_3d_final_adjusted()