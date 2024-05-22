import numpy as np
import pandas as pd
from TrainNuSVM import train_nu_svm
from TrainXGB import train_xgboost
from SplitTT import mm_test, mm_train, pb_test, pb_train
from Vote import combine_and_average_predictions
from Boom import evaluate_performance
from Plot import plot_histograms, plot_histograms_combined

# List of target column names for MM dataset
mm_columns = ['r', '0', '1', '2', '3', '4']

# List of target column names for PB dataset
pb_columns = ['r', '0', '1', '2', '3', '4']

# Train the Nu-SVM model for MM dataset
mm_predictions_nusvm = train_nu_svm((mm_train.drop(columns=['r']), mm_train[['0', '1', '2', '3', '4']]),
                                    (mm_test.drop(columns=['r']), mm_test[['0', '1', '2', '3', '4']]))

# Train the XGBoost model for MM dataset
mm_predictions_xgb = train_xgboost((mm_train.drop(columns=['r']), mm_train[['0', '1', '2', '3', '4']]),
                                   (mm_test.drop(columns=['r']), mm_test[['0', '1', '2', '3', '4']]))

# Train the Nu-SVM model for PB dataset
pb_predictions_nusvm = train_nu_svm((pb_train.drop(columns=['r']), pb_train[['0', '1', '2', '3', '4']]),
                                    (pb_test.drop(columns=['r']), pb_test[['0', '1', '2', '3', '4']]))

# Train the XGBoost model for PB dataset
pb_predictions_xgb = train_xgboost((pb_train.drop(columns=['r']), pb_train[['0', '1', '2', '3', '4']]),
                                   (pb_test.drop(columns=['r']), pb_test[['0', '1', '2', '3', '4']]))

# Combine and average predictions for MM dataset
mm_combined_predictions = combine_and_average_predictions(mm_predictions_nusvm, mm_predictions_xgb)

# Combine and average predictions for PB dataset
pb_combined_predictions = combine_and_average_predictions(pb_predictions_nusvm, pb_predictions_xgb)

# Combine MM and PB predictions
pbmmCom = mm_combined_predictions + pb_combined_predictions

# Determine the minimum number of samples between pbmmCom and pb_test_targets
min_samples = min(len(mm_combined_predictions), len(pb_combined_predictions))

# Truncate both DataFrames to the same length
pbmmCom = pbmmCom[:min_samples]

# Calculate performance metrics for the combined predictions
pbmm_accuracy, pbmm_precision, pbmm_recall, pbmm_f1 = evaluate_performance(pbmmCom, pb_test)

# Plot histograms for each target column in MM dataset for both Nu-SVM and XGBoost
plot_histograms(mm_columns, mm_predictions_nusvm, mm_predictions_xgb, mm_combined_predictions, mm_test)

# Plot histograms for each target column in PB dataset for both Nu-SVM and XGBoost
plot_histograms(pb_columns, pb_predictions_nusvm, pb_predictions_xgb, pb_combined_predictions, pb_test)

# Plot histograms for each target column in pbmmCom
plot_histograms_combined(pb_columns, pb_predictions_nusvm, pb_predictions_xgb, pbmmCom, pb_test)
