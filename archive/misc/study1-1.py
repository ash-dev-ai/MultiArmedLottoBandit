from study1 import study1
import matplotlib.pyplot as plt

def study1_1():
    # Get results from study1.py
    study1_results = study1()

    # Remove 'Total_time' key-value pair if it exists
    if 'Total_time' in study1_results:
        del study1_results['Total_time']

    # Prepare data for plotting MSE
    columns = list(study1_results.keys())
    mse_values = [study1_results[col]['Test_MSE'] for col in columns]

    # Create the plot for MSE
    plt.figure(figsize=(10, 6))
    plt.barh(columns, mse_values, color='skyblue')
    plt.xlabel('Mean Squared Error (MSE)')
    plt.title('MSE for Different Columns')
    for i, v in enumerate(mse_values):
        plt.text(v, i, f" {v:.2f}", va='center', color='black')
    plt.show()

    # Create line plots for actual vs predicted values for each column
    for column in columns:
        test_values = study1_results[column]['Test_values']
        predicted_values = study1_results[column]['Predictions']

        plt.figure(figsize=(10, 6))
        plt.plot(test_values, label='Actual', color='blue')
        plt.plot(predicted_values, label='Predicted', color='red')
        plt.title(f'Actual vs Predicted for {column}')
        plt.xlabel('Index or Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    study1_1()