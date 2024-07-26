from sklearn.model_selection import ParameterGrid
import time

# Record the start time
start_time = time.process_time()


param_grid_mnnb = {
    'alpha': [1.0, 0.5, 0.1, 0.01, 0.001]
}

param_grid_mnnb_combinations = list(ParameterGrid(param_grid_mnnb))

# Bagged Support Vector Regression (SVR)
param_grid_svr = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [1, 10, 100],
    'epsilon': [0.1, 0.01, 0.001]
}

param_grid_svr_combinations = list(ParameterGrid(param_grid_svr))

bagged_svr = None  # The actual model class and import are not needed here.

param_grid_bagged_svr = {
    'base_estimator': [bagged_svr],
    'n_estimators': [5, 10, 15]
}

param_grid_bagged_svr_combinations = list(ParameterGrid(param_grid_bagged_svr))

# Gradient Boosting Regression
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_gb_combinations = list(ParameterGrid(param_grid_gb))

# Calculate and print the elapsed time
end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time:.4f} seconds")
