import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import glob
from prophet import Prophet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ProphetModelWrapper:
    """A wrapper class for Prophet to ensure compatibility with .predict() calls in Trial0."""
    def __init__(self):
        self.model = Prophet()

    def fit(self, prophet_data):
        """Fits the Prophet model using the provided DataFrame."""
        self.model.fit(prophet_data)

    def predict(self, future_data):
        """Generates predictions using the fitted Prophet model."""
        return self.model.predict(future_data)

class MetaModel:
    def __init__(self, dataset_name, root_dir, targets=None):
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(root_dir, 'data')
        self.model_dir = os.path.join(root_dir, 'models', 'num', dataset_name)
        self.meta_model_dir = os.path.join(root_dir, 'models', 'meta', dataset_name)
        os.makedirs(self.meta_model_dir, exist_ok=True)
        self.data = self.load_data()
        self.targets = targets or ['totalSum', 'numSum']  # Default targets if none specified

    def load_data(self):
        """Loads the dataset for the specified dataset_name."""
        file_path = os.path.join(self.data_dir, f'data_{self.dataset_name}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The dataset {self.dataset_name} was not found at {file_path}")
        data = pd.read_csv(file_path)
        logging.info(f"Loaded data for {self.dataset_name} with {len(data)} rows.")
        return data

    def train_meta_models(self):
        """Trains and saves Prophet-based meta-models for each specified target."""
        logging.info(f"Training meta-models for dataset: {self.dataset_name}")
        
        for target in self.targets:
            y = self.data[target]
            prophet_data = pd.DataFrame({'ds': pd.to_datetime(self.data['draw_date']), 'y': y})

            # Instantiate and train the Prophet model
            meta_model = ProphetModelWrapper()
            meta_model.fit(prophet_data)
            
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

def main():
    logging.info("Starting meta-model training process.")
    root_dir = os.path.dirname(os.path.abspath(__file__))

    datasets = ['pb', 'mb', 'combined']
    for dataset in datasets:
        meta_model = MetaModel(dataset_name=dataset, root_dir=root_dir)
        meta_model.train_meta_models()

    logging.info("Meta-model training process completed.")

if __name__ == "__main__":
    main()
