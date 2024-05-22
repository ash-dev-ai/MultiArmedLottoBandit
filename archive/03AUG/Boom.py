import pandas as pd
import numpy as np
from TrainNuSVM import train_nu_svm
from TrainXGB import train_xgboost
from SplitTT import mm_test, mm_train, pb_test, pb_train
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def combine_and_average_predictions(predictions1, predictions2):
    combined_predictions = (predictions1 + predictions2) / 2
    return combined_predictions

def evaluate_performance(predictions, targets):
    """
    Calculate performance metrics (accuracy, precision, recall, and F1-score) for the given predictions and targets.
    """
    # Fill any NA or infinite values with 0
    predictions = predictions.fillna(0).replace([np.inf, -np.inf], 0)

    # Initialize lists to store metrics for each class
    accuracy_list, precision_list, recall_list, f1_list = [], [], [], []

    # Get the list of class labels
    class_labels = targets.columns.tolist()

    # Iterate over each class
    for class_label in class_labels:
        # Convert the predictions to integers for the current class
        predictions_int = predictions[class_label].astype(int)

        # Convert the targets to integers for the current class
        targets_int = targets[class_label].astype(int)

        # Calculate metrics for the current class
        accuracy = accuracy_score(targets_int, predictions_int)
        precision = precision_score(targets_int, predictions_int, average='macro')  # Use 'macro' for multiclass targets
        recall = recall_score(targets_int, predictions_int, average='macro')  # Use 'macro' for multiclass targets
        f1 = f1_score(targets_int, predictions_int, average='macro')  # Use 'macro' for multiclass targets

        # Append metrics to respective lists
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # Return average metrics across all classes
    return sum(accuracy_list) / len(accuracy_list), sum(precision_list) / len(precision_list), \
           sum(recall_list) / len(recall_list), sum(f1_list) / len(f1_list)
