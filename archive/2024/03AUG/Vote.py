import pandas as pd
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


if __name__ == "__main__":
    # Train the Nu-SVM model for MM dataset
    mm_train_features = mm_train.drop(columns=['r'])
    mm_train_targets = mm_train[['0', '1', '2', '3', '4']]
    mm_test_features = mm_test.drop(columns=['r'])
    mm_test_targets = mm_test[['0', '1', '2', '3', '4']]

    mm_predictions_nusvm = train_nu_svm((mm_train_features, mm_train_targets), (mm_test_features, mm_test_targets))
    mm_predictions_xgb = train_xgboost((mm_train_features, mm_train_targets), (mm_test_features, mm_test_targets))

    # Combine and average predictions for MM dataset
    mm_combined_predictions = combine_and_average_predictions(mm_predictions_nusvm, mm_predictions_xgb)

    # Calculate performance metrics for MM dataset
    mm_accuracy, mm_precision, mm_recall, mm_f1 = evaluate_performance(mm_combined_predictions, mm_test_targets)

    # Train the Nu-SVM model for PB dataset
    pb_train_features = pb_train.drop(columns=['r'])
    pb_train_targets = pb_train[['0', '1', '2', '3', '4']]
    pb_test_features = pb_test.drop(columns=['r'])
    pb_test_targets = pb_test[['0', '1', '2', '3', '4']]

    pb_predictions_nusvm = train_nu_svm((pb_train_features, pb_train_targets), (pb_test_features, pb_test_targets))
    pb_predictions_xgb = train_xgboost((pb_train_features, pb_train_targets), (pb_test_features, pb_test_targets))

    # Combine and average predictions for PB dataset
    pb_combined_predictions = combine_and_average_predictions(pb_predictions_nusvm, pb_predictions_xgb)

    # Calculate performance metrics for PB dataset
    pb_accuracy, pb_precision, pb_recall, pb_f1 = evaluate_performance(pb_combined_predictions, pb_test_targets)

    # Print the combined and averaged predictions for MM and PB datasets
    print("Combined and Averaged Predictions for MM Dataset:")
    print(mm_combined_predictions)

    print("\nCombined and Averaged Predictions for PB Dataset:")
    print(pb_combined_predictions)

    # Print performance evaluations
    print("\nPerformance Evaluation for MM Dataset:")
    print(f"Accuracy: {mm_accuracy:.4f}, Precision: {mm_precision:.4f}, Recall: {mm_recall:.4f}, F1-score: {mm_f1:.4f}")

    print("\nPerformance Evaluation for PB Dataset:")
    print(f"Accuracy: {pb_accuracy:.4f}, Precision: {pb_precision:.4f}, Recall: {pb_recall:.4f}, F1-score: {pb_f1:.4f}")
