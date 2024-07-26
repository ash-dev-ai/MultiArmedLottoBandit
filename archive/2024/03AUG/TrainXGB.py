import pandas as pd
import xgboost as xgb

def train_xgboost(train_data, test_data):
    train_features, train_targets = train_data
    test_features, test_targets = test_data

    xgb_models = {}
    predictions = pd.DataFrame()

    # If there is only one target column, convert it to a DataFrame
    if isinstance(train_targets, pd.Series):
        train_targets = pd.DataFrame({train_targets.name: train_targets})
    
    if isinstance(test_targets, pd.Series):
        test_targets = pd.DataFrame({test_targets.name: test_targets})

    # Train a separate XGBoost model for each target column
    for column in train_targets.columns:
        xgb_model = xgb.XGBRegressor()
        xgb_model.fit(train_features, train_targets[column])
        xgb_models[column] = xgb_model

    # Make predictions on the test data for all target columns
    for column, model in xgb_models.items():
        predictions[column] = model.predict(test_features)

    return predictions
