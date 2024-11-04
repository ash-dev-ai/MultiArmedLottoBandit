import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import glob
from prophet import Prophet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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
        file_path = os.path.join(self.data_dir, f'data_{self.dataset_name}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The dataset {self.dataset_name} was not found at {file_path}")
        data = pd.read_csv(file_path)
        logging.info(f"Loaded data for {self.dataset_name} with {len(data)} rows.")
        return data

    def load_trained_models(self):
        """
        Load all pre-trained individual models for the specified dataset dynamically.
        
        :return: DataFrame containing model predictions as features.
        """
        predictions = {}
        feature_columns = ['mean', 'median', 'std_dev', 'numSum', 'totalSum']
        
        # Modify path to include nested directories if needed
        model_files = glob.glob(os.path.join(self.model_dir, "**", "*.pkl"), recursive=True)
        
        if not model_files:
            logging.error("No model files found. Please check the directory and file structure.")
            return pd.DataFrame()

        for model_file in model_files:
            model_name = os.path.splitext(os.path.basename(model_file))[0]
            try:
                model = joblib.load(model_file)
                logging.info(f"Loaded model {model_name} from {model_file}")
                pred_feature_name = f"{model_name}_pred"
                
                # Assuming model expects feature_columns from self.data to make predictions
                predictions[pred_feature_name] = model.predict(self.data[feature_columns])
                
            except Exception as e:
                logging.error(f"Failed to load model {model_name} from {model_file}: {e}")

        return pd.DataFrame(predictions)

    def train_meta_models(self):
        logging.info(f"Training meta-models for dataset: {self.dataset_name}")
        
        model_predictions = self.load_trained_models()
        if model_predictions.empty:
            logging.error("No model predictions were loaded. Ensure models exist and are not empty.")
            return
        
        X = pd.concat([self.data[['mean', 'median', 'std_dev']], model_predictions], axis=1)
        
        for target in self.targets:
            y = self.data[target]
            prophet_data = pd.DataFrame({'ds': self.data['date'], 'y': y})

            # Instantiate and train the Prophet model
            meta_model = Prophet()
            meta_model.fit(prophet_data)
            
            # Make future predictions
            future = meta_model.make_future_dataframe(periods=30)  # Adjust as needed
            forecast = meta_model.predict(future)
            mse = mean_squared_error(y[-len(forecast['yhat']):], forecast['yhat'][-len(y):])
            logging.info(f"Meta-model for {target} MSE: {mse}")
            
            # Save the trained meta-model
            model_save_path = os.path.join(self.meta_model_dir, f"meta_model_{target}.pkl")
            joblib.dump(meta_model, model_save_path)
            logging.info(f"Saved meta-model for {target} to {model_save_path}")

        logging.info("Meta-model training completed for all targets.")

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
