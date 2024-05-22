from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np
import pandas as pd
from steps import calculate_steps
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Function to add binary features indicating the presence of values 1-70 in w1-w5 and pb columns
def add_value_features(df):
    for val in range(1, 71):
        new_col_name = f'val_{val}'
        df[new_col_name] = df.apply(lambda row: val in row['winning_numbers'], axis=1).astype(int)
    return df

# Create directories for saving data and visuals
current_date = datetime.now().strftime("%Y%m%d")
current_time = datetime.now().strftime('%H%M%S')

# Create folder for data if it doesn't exist
data_path = f'data/study00/{current_date}/{current_time}'
if not os.path.exists('data/study00'):
    os.makedirs('data/study00')
os.makedirs(data_path)

# Create folder for visuals if it doesn't exist
visuals_path = f'visuals/study00/{current_date}/{current_time}'
if not os.path.exists('visuals/study00'):
    os.makedirs('visuals/study00')
os.makedirs(visuals_path)

# Fetch the data
df_steps, df_steps_i = calculate_steps()

# Drop rows with NaN values for a clean run
df_steps = df_steps.dropna()
df_steps_i = df_steps_i.dropna()

# Add these new binary features to df_steps and df_steps_i
df_steps = add_value_features(df_steps)
df_steps_i = add_value_features(df_steps_i)

# Define features and targets
features = ['dw1', 'dw2', 'dw3', 'dw4', 'dw5', 'dwb', 'rw1-2', 'rw2-3', 'rw3-4', 'rw4-5', 'rwpb']
new_features = [f'val_{i}' for i in range(1, 71)]
features.extend(new_features)
targets = ['w1', 'w2', 'w3', 'w4', 'w5', 'pb']

# Initialize models
rf_model = RandomForestRegressor(n_estimators=100, random_state=137)
xgb_model = xgb.XGBRegressor(objective="reg:squarederror")

def plot_feature_importance(model, feature_names, title):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.title(title)
    plt.xlabel('Feature Importance')
    current_time = datetime.now().strftime("%Y%m%d")
    plt.savefig(f'{visuals_path}/{title}.png')
    plt.close()

def plot_residuals(y_test, pred, title):
    residuals = y_test - pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals)
    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Residuals')
    current_time = datetime.now().strftime("%Y%m%d")
    plt.savefig(f'{visuals_path}/{title}.png')
    plt.close()

def train_evaluate_model(df, name):
    print(f"\nTraining and evaluating models for DataFrame: {name}")
    for target in targets:
        print(f"Target: {target}")

        X = df[features]
        y = df[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=42)

        # Train Random Forest model
        rf_model.fit(X_train, y_train)

        # Evaluate Random Forest model
        rf_pred = rf_model.predict(X_test)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        print(f"Random Forest RMSE: {rf_rmse}")

        # Train XGBoost model
        xgb_model.fit(X_train, y_train)

        # Evaluate XGBoost model
        xgb_pred = xgb_model.predict(X_test)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        print(f"XGBoost RMSE: {xgb_rmse}")

        # Plot feature importance and residuals for RandomForest
        plot_feature_importance(rf_model, features, f'RandomForest_Feature_Importance_{name}_{target}')
        plot_residuals(y_test, rf_pred, f'RandomForest_Residuals_{name}_{target}')

        # Plot feature importance and residuals for XGBoost
        plot_feature_importance(xgb_model, features, f'XGBoost_Feature_Importance_{name}_{target}')
        plot_residuals(y_test, xgb_pred, f'XGBoost_Residuals_{name}_{target}')

# Train and evaluate for both DataFrames
train_evaluate_model(df_steps, 'df_steps')
train_evaluate_model(df_steps_i, 'df_steps_i')

def plot_feature_importance(model, feature_names, title):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.title(title)
    plt.xlabel('Feature Importance')
    plt.savefig(f'{visuals_path}/{title}.png')
    plt.close()

def plot_residuals(y_test, pred, title):
    residuals = y_test - pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals)
    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Residuals')
    plt.savefig(f'{visuals_path}/{title}.png')
    plt.close()

def train_evaluate_model(df, name):
    print(f"\nTraining and evaluating models for DataFrame: {name}")
    for target in targets:
        print(f"Target: {target}")
        
def make_and_plot_predictions(df, name):
    print(f"\nMaking and plotting predictions for DataFrame: {name}")
    for target in targets:
        print(f"Target: {target}")

        X = df[features]
        y = df[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=42)

        # Make predictions using Random Forest model
        rf_pred = rf_model.predict(X_test)
        print(f"Random Forest Predictions for {target}: {rf_pred[:10]}")  # Show first 10 predictions

        # Make predictions using XGBoost model
        xgb_pred = xgb_model.predict(X_test)
        print(f"XGBoost Predictions for {target}: {xgb_pred[:10]}")  # Show first 10 predictions

        # Plot actual vs predicted for Random Forest
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, rf_pred, color='blue', marker='o', label='Predicted')
        plt.scatter(y_test, y_test, color='red', marker='x', label='Actual')
        plt.title(f'Random Forest Actual vs Predicted for {target}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.savefig(f'{visuals_path}/RandomForest_Actual_vs_Predicted_{name}_{target}.png')
        plt.close()

        # Plot actual vs predicted for XGBoost
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, xgb_pred, color='blue', marker='o', label='Predicted')
        plt.scatter(y_test, y_test, color='red', marker='x', label='Actual')
        plt.title(f'XGBoost Actual vs Predicted for {target}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.savefig(f'{visuals_path}/XGBoost_Actual_vs_Predicted_{name}_{target}.png')
        plt.close()

# First, train and evaluate models
train_evaluate_model(df_steps, 'df_steps')
train_evaluate_model(df_steps_i, 'df_steps_i')

# Then, make and plot predictions for both DataFrames
make_and_plot_predictions(df_steps, 'df_steps')
make_and_plot_predictions(df_steps_i, 'df_steps_i')