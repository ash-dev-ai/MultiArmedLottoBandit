import matplotlib.pyplot as plt
import numpy as np
import time

# Record the start time
start_time = time.process_time()

def plot_histograms(targets, predictions_nusvm, predictions_xgb, combined_predictions, dataset_name):
    for column in targets:
        plt.figure(figsize=(75, 30))
        true_values = dataset_name['r']
        diff_nusvm = np.abs(predictions_nusvm[column] - true_values)
        diff_xgb = np.abs(predictions_xgb[column] - true_values)
        diff_combined = np.abs(combined_predictions[column] - true_values)  # Difference for combined predictions

        plt.hist(predictions_nusvm[column], bins=30, alpha=0.5, label='Nu-SVM', color='blue')
        plt.hist(predictions_xgb[column], bins=30, alpha=0.5, label='XGBoost', color='red')
        plt.hist(combined_predictions[column], bins=30, alpha=0.5, label='Combined', color='teal')  # Plot combined predictions
        plt.xlabel('Predicted Values')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'Predicted Value Distribution for "{column}" Target Column')
        plt.show()

def plot_histograms_combined(targets, predictions_nusvm, predictions_xgb, combined_predictions, dataset_name):
    for column in targets:
        plt.figure(figsize=(75, 30))
        true_values = dataset_name['r']
        diff_nusvm = np.abs(predictions_nusvm[column] - true_values)
        diff_xgb = np.abs(predictions_xgb[column] - true_values)
        diff_combined = np.abs(combined_predictions[column] - true_values)  # Difference for combined predictions
        diff_pbmm = np.abs(pbmmCom[column] - true_values)  # Difference for pbmmCom predictions

        plt.hist(predictions_nusvm[column], bins=30, alpha=0.5, label='Nu-SVM', color='blue')
        plt.hist(predictions_xgb[column], bins=30, alpha=0.5, label='XGBoost', color='red')
        plt.hist(combined_predictions[column], bins=30, alpha=0.5, label='Combined', color='teal')  # Plot combined predictions
        plt.hist(pbmmCom[column], bins=30, alpha=0.5, label='pbmmCom', color='yellow')  # Plot pbmmCom predictions from Boom
        plt.xlabel('Predicted Values')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'Predicted Value Distribution for "{column}" Target Column')
        plt.show()

# Calculate and print the elapsed time
end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time:.4f} seconds")
