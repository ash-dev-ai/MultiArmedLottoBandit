# win_main.py

import logging
import os
from win.CatBoost import CatBoostModelTrainer
from win.ElasticNet import ElasticNetModelTrainer
from win.ExtraTrees import ExtraTreesRegressorModelTrainer
from win.LightGBM import LightGBMModelTrainer
from win.MLP import MLPModelTrainer
from win.BayesianRidge import BayesianRidgeModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    logging.info("Starting the model training process.")
    
    # Determine the root directory dynamically
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Datasets to train models on
    datasets = ['pb', 'mb', 'combined']
    
    # List of trainers
    model_trainers = {
        "CatBoost": CatBoostModelTrainer,
        "ElasticNet": ElasticNetModelTrainer,
        "ExtraTrees": ExtraTreesRegressorModelTrainer,
        "LightGBM": LightGBMModelTrainer,
        "MLP": MLPModelTrainer,
        "BayesianRidge": BayesianRidgeModelTrainer
    }
    
    for dataset in datasets:
        for model_name, TrainerClass in model_trainers.items():
            logging.info(f"Training {model_name} models for {dataset}.")
            trainer = TrainerClass(dataset_name=dataset, root_dir=root_dir)
            trainer.train_models()
            logging.info(f"{model_name} models for {dataset} completed.")
    
    logging.info("Model training process completed.")

if __name__ == "__main__":
    main()


