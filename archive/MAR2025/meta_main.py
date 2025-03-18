import logging
import os
from meta.meta_model import MetaModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    logging.info("Starting the meta-model training process.")
    
    # Define the root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the datasets
    datasets = ['pb', 'mb', 'combined']
    
    # Initialize and train meta-models for each dataset
    for dataset in datasets:
        logging.info(f"Creating {dataset.upper()} meta model.")
        
        # Initialize the MetaModel with the dataset name and root directory
        meta_model = MetaModel(dataset_name=dataset, root_dir=root_dir)
        
        # Train and save the meta-models
        meta_model.train_meta_models()
        
        logging.info(f"{dataset.upper()} meta model completed.")
    
    logging.info("Meta-model training process completed.")

if __name__ == "__main__":
    main()
