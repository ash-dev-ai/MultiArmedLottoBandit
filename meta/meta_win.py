import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import glob
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class MetaWinModel:
    def __init__(self, dataset_name, root_dir, targets=None):
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(root_dir, 'data')
        self.model_dir = os.path.join(root_dir, 'models', 'win', dataset_name)
        self.meta_model_dir = os.path.join(root_dir, 'models', 'meta', 'win', dataset_name)
        os.makedirs(self.meta_model_dir, exist_ok=True)
        self.data = self.load_data()
        self.targets = targets or ['mean', 'median', 'totalSum']  # Example targets for the "win" setup

    def load_data(self):
        """Loads the dataset for the specified dataset_name."""
        file_path = os.path.join(self.data_dir, f'data_{self.dataset_name}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The dataset {self.dataset_name} was not found at {file_path}")
        data = pd.read_csv(file_path)
        logging.info(f"Loaded data for {self.dataset_name} with {len(data)} rows.")
        return data

    def train_meta_models(self):
        """Trains and saves RandomForest-based meta-models for each specified target."""
        logging.info(f"Training meta-models for dataset: {self.dataset_name}")
        
        for target in ['num1', 'num2', 'num3', 'num4', 'num5', 'numA']:
            y = self.data[target]
            
            # Select all numeric columns except the current target and drop 'draw_date' and 'winning_numbers'
            X = self.data.drop(columns=['draw_date', 'winning_numbers', target]).select_dtypes(include=['float64', 'int64'])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Instantiate and train the RandomForest model
            meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
            meta_model.fit(X_train, y_train)
            
            # Evaluate and log performance
            predictions = meta_model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            logging.info(f"Meta-model for {target}: MSE={mse:.4f}")
            
            # Save the trained meta-model
            model_save_path = os.path.join(self.meta_model_dir, f"meta_model_{target}.pkl")
            joblib.dump(meta_model, model_save_path)
            logging.info(f"Saved meta-model for {target} to {model_save_path}")
    
        logging.info("Meta-model training completed for all targets.")

    def load_trained_models(self):
        """Loads the trained meta-models."""
        meta_models = {}
        for target in self.targets:
            model_path = os.path.join(self.meta_model_dir, f"meta_model_{target}.pkl")
            if os.path.exists(model_path):
                meta_models[target] = joblib.load(model_path)
                logging.info(f"Loaded meta-model for {target} from {model_path}")
            else:
                logging.warning(f"Meta-model for {target} not found at {model_path}")
        return meta_models
