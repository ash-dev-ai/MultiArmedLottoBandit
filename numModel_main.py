import logging
import os
from nums.gbc import GBCModelTrainer
from nums.rfc import RFCModelTrainer
from nums.svm import SVMModelTrainer
from nums.logreg import LogRegModelTrainer
from nums.knn import KNNModelTrainer
from nums.adaboost import AdaBoostModelTrainer
from nums.decision_tree import DecisionTreeModelTrainer
from nums.naive_bayes import NaiveBayesModelTrainer
from nums.svc import SVCModelTrainer
from nums.extra_trees import ExtraTreesModelTrainer
from nums.gbr import GBRModelTrainer
from nums.xgboost import XGBoostModelTrainer
from nums.lightgbm import LightGBMModelTrainer  # Import the LightGBM model trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    logging.info("Starting the model training process.")

    # Determine the root directory dynamically
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize and train models for Powerball (pb), Mega Millions (mb), and Combined (comb) datasets
    datasets = ['pb', 'mb', 'combined']
    
    for dataset in datasets:
        # Train GBC models
        logging.info(f"Training GBC models for {dataset}.")
        gbc_trainer = GBCModelTrainer(dataset_name=dataset, root_dir=root_dir)
        gbc_trainer.train_models()
        logging.info(f"GBC models for {dataset} completed.")
        
        # Train RFC models
        logging.info(f"Training RFC models for {dataset}.")
        rfc_trainer = RFCModelTrainer(dataset_name=dataset, root_dir=root_dir)
        rfc_trainer.train_models()
        logging.info(f"RFC models for {dataset} completed.")
        
        # Train SVM models
        logging.info(f"Training SVM models for {dataset}.")
        svm_trainer = SVMModelTrainer(dataset_name=dataset, root_dir=root_dir)
        svm_trainer.train_models()
        logging.info(f"SVM models for {dataset} completed.")
        
        # Train Logistic Regression models
        logging.info(f"Training Logistic Regression models for {dataset}.")
        logreg_trainer = LogRegModelTrainer(dataset_name=dataset, root_dir=root_dir)
        logreg_trainer.train_models()
        logging.info(f"Logistic Regression models for {dataset} completed.")
        
        # Train KNN models
        logging.info(f"Training KNN models for {dataset}.")
        knn_trainer = KNNModelTrainer(dataset_name=dataset, root_dir=root_dir)
        knn_trainer.train_models()
        logging.info(f"KNN models for {dataset} completed.")
        
        # Train AdaBoost models
        logging.info(f"Training AdaBoost models for {dataset}.")
        adaboost_trainer = AdaBoostModelTrainer(dataset_name=dataset, root_dir=root_dir)
        adaboost_trainer.train_models()
        logging.info(f"AdaBoost models for {dataset} completed.")
        
        # Train Decision Tree models
        logging.info(f"Training Decision Tree models for {dataset}.")
        decision_tree_trainer = DecisionTreeModelTrainer(dataset_name=dataset, root_dir=root_dir)
        decision_tree_trainer.train_models()
        logging.info(f"Decision Tree models for {dataset} completed.")
        
        # Train Naive Bayes models
        logging.info(f"Training Naive Bayes models for {dataset}.")
        naive_bayes_trainer = NaiveBayesModelTrainer(dataset_name=dataset, root_dir=root_dir)
        naive_bayes_trainer.train_models()
        logging.info(f"Naive Bayes models for {dataset} completed.")
        
        # Train SVC models
        logging.info(f"Training SVC models for {dataset}.")
        svc_trainer = SVCModelTrainer(dataset_name=dataset, root_dir=root_dir)
        svc_trainer.train_models()
        logging.info(f"SVC models for {dataset} completed.")
        
        # Train Extra Trees models
        logging.info(f"Training Extra Trees models for {dataset}.")
        extra_trees_trainer = ExtraTreesModelTrainer(dataset_name=dataset, root_dir=root_dir)
        extra_trees_trainer.train_models()
        logging.info(f"Extra Trees models for {dataset} completed.")
        
        # Train Gradient Boosting Regressor models
        logging.info(f"Training Gradient Boosting Regressor models for {dataset}.")
        gbr_trainer = GBRModelTrainer(dataset_name=dataset, root_dir=root_dir)
        gbr_trainer.train_models()
        logging.info(f"Gradient Boosting Regressor models for {dataset} completed.")
        
        # Train XGBoost models
        logging.info(f"Training XGBoost models for {dataset}.")
        xgboost_trainer = XGBoostModelTrainer(dataset_name=dataset, root_dir=root_dir)
        xgboost_trainer.train_models()
        logging.info(f"XGBoost models for {dataset} completed.")

        # Train LightGBM models
        logging.info(f"Training LightGBM models for {dataset}.")
        lightgbm_trainer = LightGBMModelTrainer(dataset_name=dataset, root_dir=root_dir)
        lightgbm_trainer.train_models()
        logging.info(f"LightGBM models for {dataset} completed.")

    logging.info("Model training process completed.")

if __name__ == "__main__":
    main()
