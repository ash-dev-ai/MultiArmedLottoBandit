# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

class WinModelTrainer:
    def __init__(self, dataset_name, root_dir):
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(root_dir, 'data')
        self.model_dir = os.path.join(root_dir, 'models', 'win', dataset_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.data = self.load_data()
        self.models = {}  # To store model instances if needed dynamically

    def load_data(self):
        """Loads the dataset into a pandas DataFrame."""
        file_path = os.path.join(self.data_dir, f'data_{self.dataset_name}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The dataset {self.dataset_name} was not found at {file_path}")
        return pd.read_csv(file_path)

    def create_target_column(self, target_column_name):
        """
        Creates the target column for binary classification.
        For num1-num5, the target is 1 if the number matches, 0 otherwise.
        """
        if target_column_name in ['num1', 'num2', 'num3', 'num4', 'num5']:
            # Create a binary target for num1-num5
            return self.data[target_column_name].astype(int)
        elif target_column_name == 'numA':
            # Binary target for numA
            return self.data[target_column_name].astype(int)
        else:
            raise ValueError(f"Invalid target column {target_column_name}.")

    def augment_data(self, X, y, num):
        """Augments data to ensure all possible numbers are represented."""
        unique_classes = y.unique()
        if len(unique_classes) == 1:  # Only one class present, add the missing class
            synthetic_X = X.sample(n=1, random_state=42)
            synthetic_y = pd.Series([1 - unique_classes[0]])  # Add the opposite class (binary targets)
            X = pd.concat([X, synthetic_X], ignore_index=True)
            y = pd.concat([y, synthetic_y], ignore_index=True)
        return X, y

    def train_and_save_model(self, model, model_name, target_column):
        """Trains the model and saves it."""
        features = self.data[['mean', 'median', 'std_dev', 'numSum', 'totalSum']]  # Example feature set
        target = self.data[target_column]
    
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
        # Handle target column with non-numeric suffix
        if target_column.startswith('num'):
            num = target_column[3:]  # Extract numeric part or 'A' from 'num1', 'num2', ..., 'numA'
        else:
            raise ValueError(f"Invalid target column {target_column}")
    
        if num.isdigit():  # Only augment if it is numeric (e.g., '1', '2', etc.)
            X_train, y_train = self.augment_data(X_train, y_train, num=int(num))
    
        # Train the model
        model.fit(X_train, y_train)
        model_output_path = os.path.join(self.model_dir, f'{model_name}.pkl')
        joblib.dump(model, model_output_path)

    def train_all_targets(self):
        """Train models for all targets (num1 to num5 and numA)."""
        target_columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'numA']
        for target_column in target_columns:
            model_name = f"{self.__class__.__name__}_{target_column}"  # Standardized naming
            self.data[f"{target_column}_target"] = self.create_target_column(target_column)  # Create target
            model = self.initialize_model()  # Model-specific initialization
            self.train_and_save_model(model, model_name, f"{target_column}_target")
