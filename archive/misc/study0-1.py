import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.stats import shapiro
from scipy.fft import fft, fftfreq

# Load the checkpoint data
def load_checkpoint(checkpoint_file="checkpoint.pkl"):
    try:
        with open(checkpoint_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}, {}

all_fits, all_results = load_checkpoint()

def plot_results(all_results):
    for dataset_label, column_results in all_results.items():
        print(f"--- Plotting results for Label: {dataset_label} ---")
        for column, result_list in column_results.items():
            for result in result_list:
                column_values = result['column_values']

                # Pearson Correlation Plot
                plt.figure()
                plt.scatter(column_values, result['predictions'])
                plt.title(f'Pearson Correlation Between Test and Predictions for Column: {column}')
                plt.xlabel('Test Data Points')
                plt.ylabel('Prediction Data Points')
                plt.show()

                # Shapiro-Wilk Normality Test Results
                print(f"Shapiro-Wilk Test Statistic: {result['Shapiro-Wilk_test_statistic']}")
                print(f"Shapiro-Wilk p-value: {result['Shapiro-Wilk_p_value']}")

                # Bayesian Confidence Histogram
                plt.figure()
                plt.hist(result['column_residuals'], bins=50, density=True, alpha=0.0729, color="g")
                plt.axvline(result['Bayesian_mean'], color='red', linestyle='dashed', linewidth=2)
                plt.title(f'Bayesian Confidence Interval for Residuals - {column}')
                plt.show()

                # Fourier Transform Plot
                N = len(column_values)
                if N > 0:  # Only proceed if N is greater than zero
                    T = 1.0  # Time interval
                    x = np.linspace(0.0, N * T, N, endpoint=False)
                    yf = result['Fourier_terms']
                    xf = fftfreq(N, T)[:N // 2]
                    plt.figure()
                    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
                    plt.title(f'Fourier Transform of Test Data for Column: {column}')
                    plt.show()
                else:
                    print(f"Skipping Fourier Transform Plot for {column} due to lack of data.")

                # Histogram of Test Data
                plt.figure()
                plt.hist(column_values, bins=50, density=True, alpha=0.5, color="blue")
                plt.title(f'Histogram of Test Data for Column: {column}')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.show()

                # Time Series Plot of Predictions and Test Data
                plt.figure()
                plt.plot(result['predictions'], label='Predicted Values', color='green')
                plt.plot(column_values, label='Actual Test Values', color='blue')
                plt.title(f'Time Series of Predictions vs Actuals for Column: {column}')
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.legend()
                plt.show()

def plot_all_fits(all_fits):
    for dataset_label, fits in all_fits.items():
        print(f"--- Plotting fits for Label: {dataset_label} ---")
        for column, model_info in fits.items():
            plt.figure()
            plt.plot(model_info['model'].fittedvalues, label='Fitted Values')
            plt.title(f"Fitted values for {dataset_label}, column {column}")
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.show()

if __name__ == '__main__':
    plot_results(all_results)
    plot_all_fits(all_fits)


