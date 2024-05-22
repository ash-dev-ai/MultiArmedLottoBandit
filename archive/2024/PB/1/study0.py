from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from steps import calculate_steps
import os

# Create directories for saving data and visuals
current_date = datetime.now().strftime("%Y%m%d")
current_time = datetime.now().strftime('%H%M%S')

# Create a folder named 'data' if it doesn't exist
data_path = f'data/study0/{current_date}/{current_time}'
if not os.path.exists('data/study0'):
    os.makedirs('data/study0')
os.makedirs(data_path)

# Create a folder named 'visuals/study0' if it doesn't exist
visuals_path = f'visuals/study0/{current_date}/{current_time}'
if not os.path.exists('visuals/study0'):
    os.makedirs('visuals/study0')
os.makedirs(visuals_path)

def save_to_csv(df, name):
    df.to_csv(f'{data_path}/{name}.csv')

def generate_and_save_heatmap(df, title, filename):
    plt.figure(figsize=(16, 12))
    sns.heatmap(df, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(title)
    plt.savefig(f'{visuals_path}/{filename}')
    plt.close()

def generate_and_save_boxplot(df, title, filename):
    plt.figure(figsize=(16, 12))
    sns.boxplot(data=df)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.savefig(f'{visuals_path}/{filename}')
    plt.close()

def plot_descriptive_stats_row_by_row(df, df_name):
    for row in df.index:
        generate_and_save_boxplot(df.loc[[row]], f"Boxplot for {row} of {df_name}", f"{df_name}_{row}_boxplot.png")

def main():
    df_steps, df_steps_i = calculate_steps()

    # Descriptive Statistics
    df_steps_desc_stats = df_steps.describe()
    df_steps_i_desc_stats = df_steps_i.describe()
    save_to_csv(df_steps_desc_stats, 'df_steps_desc_stats')
    save_to_csv(df_steps_i_desc_stats, 'df_steps_i_desc_stats')

    # Correlation Analysis
    df_steps_corr_matrix = df_steps.corr()
    df_steps_i_corr_matrix = df_steps_i.corr()
    save_to_csv(df_steps_corr_matrix, 'df_steps_corr_matrix')
    save_to_csv(df_steps_i_corr_matrix, 'df_steps_i_corr_matrix')

    # Heatmaps
    generate_and_save_heatmap(df_steps_corr_matrix, "Correlation Matrix Heatmap for df_steps", "df_steps_corr_matrix_heatmap.png")
    generate_and_save_heatmap(df_steps_i_corr_matrix, "Correlation Matrix Heatmap for df_steps_i", "df_steps_i_corr_matrix_heatmap.png")

    # Boxplots
    plot_descriptive_stats_row_by_row(df_steps_desc_stats, 'df_steps')
    plot_descriptive_stats_row_by_row(df_steps_i_desc_stats, 'df_steps_i')

if __name__ == '__main__':
    main()