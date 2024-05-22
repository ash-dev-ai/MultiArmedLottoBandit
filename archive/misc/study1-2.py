import numpy as np
import matplotlib.pyplot as plt
from study1 import study1  # Assuming study1 function is imported from study1.py

alpha = 1 / 137  # Fine-structure constant
beta = 2 * np.pi  # Tau

def quantum_interference(value, alpha, beta):
    # Implementing the 2n-slit quantum interference pattern
    return value * (1 + alpha * np.sin(beta * value))

def study1_2():
    study1_results = study1()

    # Create a new dictionary to store the interfered predictions
    interfered_results = {}

    for column, results in study1_results.items():
        if column == 'Total_time':  # Skip the total time entry
            continue

        predictions = results['Predictions']
        
        # Apply quantum interference to the predictions
        interfered_predictions = quantum_interference(np.array(predictions), alpha, beta)

        # Create a plot
        plt.figure(figsize=(14, 6))
        plt.plot(results['Test_values'], label='Actual')
        plt.plot(interfered_predictions, label='Interfered Prediction')
        plt.title(f"{column} - Actual vs Interfered Prediction")
        plt.legend()
        plt.show()

        # Store the interfered predictions
        interfered_results[column] = {'Interfered_Predictions': interfered_predictions}

    return interfered_results

if __name__ == "__main__":
    study1_2_results = study1_2()
    print(f"Interfered Results: {study1_2_results}")