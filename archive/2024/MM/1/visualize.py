import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def load_latest_dataset():
    base_path = 'data/prep'
    latest_date = sorted(os.listdir(base_path), reverse=True)[0]
    latest_time = sorted(os.listdir(f"{base_path}/{latest_date}"), reverse=True)[0]
    latest_path = f"{base_path}/{latest_date}/{latest_time}/org_df.csv"
    return pd.read_csv(latest_path)

def parse_sum_and_mega_ball(string):
    cleaned_string = string.strip("[]")
    numbers = cleaned_string.split(", ")
    return [int(numbers[0]), int(numbers[1])]

def perform_eda(df):
    # Assuming visuals_path is defined elsewhere in visualize.py or passed to this function
    visuals_path = 'visuals/prep'  # Modify as needed
    current_date = datetime.now().strftime("%Y%m%d")
    current_time = datetime.now().strftime('%H%M%S')
    visuals_specific_path = f'{visuals_path}/{current_date}/{current_time}'
    os.makedirs(visuals_specific_path, exist_ok=True)

    df[['w1', 'w2', 'w3', 'w4', 'w5', 'mega_ball']].hist(bins=200, figsize=(50, 30))
    plt.savefig(f"{visuals_specific_path}/number_distributions.png")
    plt.show()

    # Correlation matrix
    correlation_matrix = df[['w1', 'w2', 'w3', 'w4', 'w5', 'mega_ball']].corr()
    print(correlation_matrix)


def visualize_data(df):
    # Convert 'draw_date' to datetime and 'd' to int
    df['draw_date'] = pd.to_datetime(df['draw_date'])
    df['d'] = df['d'].astype(int)

    # Set 'draw_date' as the index
    df.set_index('draw_date', inplace=True)

    # Plot total_sum over d for all days
    plt.figure(figsize=(200, 50))
    plt.plot(df['d'], df['total_sum'], marker='o', linestyle='-', label='Overall')
    plt.title('Total Sum over Draw Number (Overall)')
    plt.xlabel('Draw Number (d)')
    plt.ylabel('Total Sum')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot total_sum over d for Tuesdays
    tuesday_df = df[df['weekday'] == 'Tuesday']
    plt.figure(figsize=(200, 50))
    plt.plot(tuesday_df['d'], tuesday_df['total_sum'], marker='o', color='green', linestyle='-', label='Tuesday')
    plt.title('Total Sum over Draw Number (Tuesday)')
    plt.xlabel('Draw Number (d)')
    plt.ylabel('Total Sum')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot total_sum over d for Fridays
    friday_df = df[df['weekday'] == 'Friday']
    plt.figure(figsize=(200, 50))
    plt.plot(friday_df['d'], friday_df['total_sum'], marker='o', color='red', linestyle='-', label='Friday')
    plt.title('Total Sum over Draw Number (Friday)')
    plt.xlabel('Draw Number (d)')
    plt.ylabel('Total Sum')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    df = load_latest_dataset()
    perform_eda(df)
    visualize_data(df)

if __name__ == '__main__':
    main()