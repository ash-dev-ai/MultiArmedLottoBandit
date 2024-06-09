# reservoir_ensemble.py
import numpy as np
import pandas as pd
from datetime import datetime
import os
import logging
import pickle
from sklearn.linear_model import Ridge
from stacked.libs.pyESN.pyESN import ESN
from reservoirpy.nodes import Reservoir, Ridge as LSMRidge  # Liquid State Machine implementation

def prepare_reservoir_data(data):
    """Prepare features for the reservoir models."""
    # Select the relevant features
    X = data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA', 'numSum', 'totalSum', 'day', 'date']].copy()
    y = data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].copy()

    # Normalize the features
    X = (X - X.mean()) / X.std()

    return X.values, y.values

def train_esn_model(X_train, y_train, X_val, y_val, reservoir_size=500):
    """Train an Echo State Network (ESN) model."""
    esn = ESN(
        n_inputs=X_train.shape[1],
        n_outputs=y_train.shape[1],
        n_reservoir=reservoir_size,
        sparsity=0.2,
        random_state=42,
        spectral_radius=0.95
    )
    
    # Fit the ESN without the alpha parameter
    esn.fit(X_train, y_train)
    return esn

def train_lsm_model(X_train, y_train, X_val, y_val, reservoir_size=500):
    """Train a Liquid State Machine (LSM) model."""
    # Initialize the reservoir with the number of units
    reservoir = Reservoir(units=reservoir_size)
    
    # Set the spectral radius for the reservoir
    reservoir.spectral_radius = 0.95
    
    # Create a pipeline with the reservoir followed by a ridge regression (no alpha)
    ridge = LSMRidge()  # Ridge initialization without alpha
    lsm = reservoir >> ridge
    
    # Fit the model
    lsm = lsm.fit(X_train, y_train)
    return lsm

def train_reservoir_ensemble(train_data, val_data, dataset_type):
    X_train, y_train = prepare_reservoir_data(train_data)
    X_val, y_val = prepare_reservoir_data(val_data)

    # Train ESN model
    esn_model = train_esn_model(X_train, y_train, X_val, y_val)

    # Train LSM model
    lsm_model = train_lsm_model(X_train, y_train, X_val, y_val)

    models = {
        'esn': esn_model,
        'lsm': lsm_model
    }

    save_models(models, f'stacked/models')

    return models

def evaluate_reservoir_ensemble(models, test_data):
    X_test, _ = prepare_reservoir_data(test_data)
    predictions = {name: model.predict(X_test) if name == 'esn' else model.run(X_test) for name, model in models.items()}
    return predictions

def save_models(models, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for name, model in models.items():
        if name == 'esn':
            # Save ESN model parameters
            np.save(os.path.join(directory, f'{name}_W_in.npy'), model.W_in)
            np.save(os.path.join(directory, f'{name}_W.npy'), model.W)
            np.save(os.path.join(directory, f'{name}_W_out.npy'), model.W_out)
        elif name == 'lsm':
            # Save LSM model using pickle
            with open(os.path.join(directory, f'{name}_model.pkl'), 'wb') as f:
                pickle.dump(model, f)

def load_models(directory):
    models = {}
    # Load ESN model parameters
    esn_W_in = np.load(os.path.join(directory, 'esn_W_in.npy'))
    esn_W = np.load(os.path.join(directory, 'esn_W.npy'))
    esn_W_out = np.load(os.path.join(directory, 'esn_W_out.npy'))
    
    # Reconstruct the ESN model (you will need the exact architecture parameters)
    esn_model = ESN(
        n_inputs=esn_W_in.shape[1],
        n_outputs=esn_W_out.shape[0],
        n_reservoir=esn_W.shape[0]
    )
    esn_model.W_in = esn_W_in
    esn_model.W = esn_W
    esn_model.W_out = esn_W_out
    models['esn'] = esn_model

    # Load LSM model using pickle
    with open(os.path.join(directory, 'lsm_model.pkl'), 'rb') as f:
        lsm_model = pickle.load(f)
    models['lsm'] = lsm_model
    
    return models

def save_predictions(predictions, dataset_name, dataset_type):
    today = datetime.today().strftime('%Y-%m-%d')
    predictions_dir = 'data/predictions'
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    
    for model_name, preds in predictions.items():
        predictions_df = pd.DataFrame(preds, columns=['num1', 'num2', 'num3', 'num4', 'num5', 'numA'])
        predictions_file = os.path.join(predictions_dir, f'{model_name}_{dataset_type}_predictions_{today}.csv')
        predictions_df.to_csv(predictions_file, index=False)
        logging.info(f"Predictions saved to {predictions_file}")

def main():
    # Load datasets
    train_combined = pd.read_csv('data/train_combined.csv')
    val_combined = pd.read_csv('data/val_combined.csv')
    test_combined = pd.read_csv('data/test_combined.csv')

    train_pb = pd.read_csv('data/train_pb.csv')
    val_pb = pd.read_csv('data/val_pb.csv')
    test_pb = pd.read_csv('data/test_pb.csv')

    train_mb = pd.read_csv('data/train_mb.csv')
    val_mb = pd.read_csv('data/val_mb.csv')
    test_mb = pd.read_csv('data/test_mb.csv')

    # Train and evaluate Reservoir ensemble for each dataset type
    logging.info("Training Reservoir ensemble for combined dataset...")
    models_combined = train_reservoir_ensemble(train_combined, val_combined, 'combined')
    predictions_combined = evaluate_reservoir_ensemble(models_combined, test_combined)
    save_predictions(predictions_combined, 'combined', 'combined')

    logging.info("Training Reservoir ensemble for PB dataset...")
    models_pb = train_reservoir_ensemble(train_pb, val_pb, 'pb')
    predictions_pb = evaluate_reservoir_ensemble(models_pb, test_pb)
    save_predictions(predictions_pb, 'pb', 'pb')

    logging.info("Training Reservoir ensemble for MB dataset...")
    models_mb = train_reservoir_ensemble(train_mb, val_mb, 'mb')
    predictions_mb = evaluate_reservoir_ensemble(models_mb, test_mb)
    save_predictions(predictions_mb, 'mb', 'mb')

    logging.info("Predictions for combined dataset:")
    print(predictions_combined)

    logging.info("Predictions for PB dataset:")
    print(predictions_pb)

    logging.info("Predictions for MB dataset:")
    print(predictions_mb)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
