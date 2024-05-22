import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from steps import calculate_steps
from prep import org_df
from study00 import df
from datetime import datetime

# At the beginning of the script
steps_df, _, _ = calculate_steps()

def add_arima_features_to_X(X, df, keys_to_extract):
    for index, row in df.iterrows():
        column = row['Column']  # Extract the 'Column' information
        results_data = row['Results Data']  # Extract the 'Results Data' list
        
        if isinstance(results_data, list):
            for item in results_data:
                if isinstance(item, dict):
                    new_col_prefix = f"{column}_"
                    for key, value in item.items():
                        if key in keys_to_extract:
                            new_col_name = f"{new_col_prefix}{key}"
                            X[new_col_name] = np.nan  # Initialize new column with NaNs
                            if hasattr(value, '__len__'):  # Array-like values
                                X.loc[X.index[-len(value):], new_col_name] = value
                            else:  # Scalar values
                                X[new_col_name] = value
    return X

def prepare_data(org_df):
    org_df_filtered = org_df.iloc[1:]
    X = steps_df[['dw1', 'dw2', 'dw3', 'dw4', 'dw5', 'dwb', 'rw1-2', 'rw2-3', 'rw3-4', 'rw4-5', 'rwpb']]
    Y = org_df_filtered[['w1', 'w2', 'w3', 'w4', 'w5', 'pb']]
    return X, Y

def update_org_df(org_df, pred_row):
    last_row = org_df.iloc[-1]
    new_row = last_row + pred_row
    return org_df.append(new_row, ignore_index=True)

def make_future_predictions(models, X, Y, org_df):
    future_preds = {}
    X, Y = prepare_data(org_df)
    for model_name, model in models.items():
        future_preds[model_name] = []
        for _ in range(10):
            last_X = X.iloc[-1:].values
            next_Y_pred = model.predict(last_X)
            next_Y_pred = np.round_(next_Y_pred, 0)
            future_preds[model_name].append(next_Y_pred[0])
            pred_row = pd.Series(next_Y_pred[0], index=Y.columns)
            org_df = update_org_df(org_df, pred_row)
            X, Y = prepare_data(org_df)
    return future_preds, org_df

def train_evaluate_linear_regression(X_train, Y_train, X_test, Y_test):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, Y_train)
    Y_pred = lin_reg.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    print(f"Linear Regression MSE: {mse}")
    return lin_reg

def train_evaluate_random_forest(X_train, Y_train, X_test, Y_test):
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train, Y_train)
    Y_pred = rf_reg.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    print(f"Random Forest MSE: {mse}")
    return rf_reg

def train_evaluate_xgboost(X_train, Y_train, X_test, Y_test):
    xgb_reg = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    xgb_reg.fit(X_train, Y_train)
    Y_pred = xgb_reg.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    print(f"XGBoost MSE: {mse}")
    return xgb_reg

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Script started at: {start_time}")

    X, Y = prepare_data(org_df)
    keys_to_extract = ['Pearson_correlation', 'Shapiro-Wilk_test_statistic', 'Bayesian_mean']
    X_with_arima = add_arima_features_to_X(X.copy(), df, keys_to_extract)
    original_X = X[['dw1', 'dw2', 'dw3', 'dw4', 'dw5', 'dwb', 'rw1-2', 'rw2-3', 'rw3-4', 'rw4-5', 'rwpb']]
    X_train, X_test, Y_train, Y_test = train_test_split(original_X, Y, test_size=0.27, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42),  # Adjust hyperparameters
        "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.00729927007 )  # Adjust hyperparameters
    }

    for name, model in models.items():
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        print(f"{name} MSE: {mse}")

    future_preds, org_df_updated = make_future_predictions(models, X_with_arima, Y, org_df)
    future_preds_df = pd.DataFrame.from_dict(future_preds, orient='index').transpose()

    print(future_preds_df)

    end_time = datetime.now()
    print(f"Script stopped at: {end_time}")
    print(f"Total time elapsed: {end_time - start_time}")