from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adadelta
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from steps import calculate_steps
from tensorflow.keras.models import load_model

# Function to add binary features indicating the presence of values 1-70 in w1-w5 and pb columns
def add_value_features(df):
    for val in range(1, 71):
        new_col_name = f'val_{val}'
        df[new_col_name] = df.apply(lambda row: val in row['winning_numbers'], axis=1).astype(int)
    return df

# Create directories for saving data and visuals
current_date = datetime.now().strftime("%Y%m%d")
current_time = datetime.now().strftime('%H%M%S')

# Create a folder named 'data' if it doesn't exist
data_path = f'data/study000/{current_date}/{current_time}'
if not os.path.exists('data/study000'):
    os.makedirs('data/study000')
os.makedirs(data_path)

# Create a folder named 'visuals/study000' if it doesn't exist
visuals_path = f'visuals/study000/{current_date}/{current_time}'
if not os.path.exists('visuals/study000'):
    os.makedirs('visuals/study000')
os.makedirs(visuals_path)

# Fetch the data and drop NaN rows
df_steps, df_steps_i = calculate_steps()
df_steps = df_steps.dropna()
df_steps_i = df_steps_i.dropna()

# Add these new binary features to df_steps and df_steps_i
df_steps = add_value_features(df_steps)
df_steps_i = add_value_features(df_steps_i)

# Define features and targets
features = ['dw1', 'dw2', 'dw3', 'dw4', 'dw5', 'dwb', 'rw1-2', 'rw2-3', 'rw3-4', 'rw4-5', 'rwpb']
new_features = [f'val_{i}' for i in range(1, 71)]
all_features = features + new_features  # Combining original and new features
targets = ['w1', 'w2', 'w3', 'w4', 'w5', 'pb']


# Create adjusted features
#klfeatures = ['dw1', 'dw2', 'dw3', 'dw4', 'dw5', 'dwb']
adjusts = ['rw1-2', 'rw2-3', 'rw3-4', 'rw4-5', 'rwpb']
for feature, adjust in zip(features[1:], adjusts):  # Skip 'dw1' as it does not adjust
    df_steps[f'adjusted_{feature}'] = df_steps[feature] + df_steps[adjust]
    df_steps_i[f'adjusted_{feature}'] = df_steps_i[feature] + df_steps_i[adjust]

# Your new feature list becomes:
new_features = ['dw1', 'adjusted_dw2', 'adjusted_dw3', 'adjusted_dw4', 'adjusted_dw5', 'adjusted_dwb']

# Function to apply Gaussian Processes
def apply_gaussian_processes(X_train, y_train, X_test, y_test):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X_train, y_train)
    y_pred, sigma = gp.predict(X_test, return_std=True)
    return y_pred, sigma

# Function to apply t-SNE
def apply_tsne(X):
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

# New function to train Lorenz Surrogate Model
def train_lorenz_surrogate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=37)
    return model, scaler

# New function to plot Lorenz surrogate results
def plot_lorenz_results(y_test, pred, target, d_values, df_name, visuals_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(d_values, pred, c='purple', label='Lorenz Predicted')
    plt.scatter(d_values, y_test, c='orange', label='Actual')
    plt.title(f'Lorenz Surrogate Actual vs Predicted for {df_name}_{target}')
    plt.xlabel('d')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.savefig(f'{visuals_path}/{df_name}_{target}_Lorenz_results.png')
    plt.close()

# Function to plot LSTM results
def plot_lstm_results(y_test, pred, target, d_values, df_name, visuals_path):
    plt.figure(figsize=(15, 6))
    
    # Plot actual vs predicted
    plt.subplot(1, 2, 1)
    plt.scatter(d_values, pred, c='green', label='Predicted')
    plt.scatter(d_values, y_test, c='red', label='Actual')
    plt.title(f'Actual vs Predicted for {df_name}_{target}')
    plt.xlabel('d')
    plt.ylabel('Value')
    plt.legend(loc='best')

    # Plot residuals by d
    residuals = y_test - pred
    plt.subplot(1, 2, 2)
    plt.scatter(d_values, residuals, c='blue', label='Residuals')
    plt.title(f'Residuals for {df_name}_{target} by d')
    plt.xlabel('d')
    plt.ylabel('Residual')
    plt.axhline(0, color='red', linestyle='--')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f'{visuals_path}/{df_name}_{target}_results.png')
    plt.close()

# Function to plot Gaussian Process results
def plot_gp_results(y_test, gp_pred, gp_sigma, target, df_name, visuals_path):    
    # Plot actual vs predicted
    plt.scatter(d_values, gp_pred, c='green', label='GP Predicted')
    plt.scatter(d_values, y_test, c='red', label='Actual')
    
    # Plot confidence intervals
    plt.fill_between(y_test.flatten(), gp_pred - gp_sigma, gp_pred + gp_sigma, color='lightblue', alpha=0.5)
    
    plt.title(f'Gaussian Process Predictions with Confidence Intervals for {target}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.savefig(f'{visuals_path}/{df_name}_{target}_GP_results.png')
    plt.close()
    
# Function to plot t-SNE results
def plot_tsne_results(X_tsne, y, target, df_name, visuals_path):
    # 2D Scatter plot for t-SNE
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')  # Color by target variable for better visibility
    plt.title(f't-SNE plot for {df_name}_{target}')
    plt.colorbar(label='Target Variable')
    plt.savefig(f'{visuals_path}/{df_name}_{target}_tSNE_plot.png')
    plt.close()

data_frames = {'df_steps': df_steps, 'df_steps_i': df_steps_i}

loss_options = ['mean_squared_error', 'mean_squared_logarithmic_error', 'huber']

for loss_option in loss_options:
    print(f"Using loss function: {loss_option}")
    for df_name, df in data_frames.items():
        for target in targets:
            feature_scaler = MinMaxScaler(feature_range=(-1, 1))
            target_scaler = MinMaxScaler(feature_range=(-1, 1))
    
            X = df[all_features]
            y = df[target]
            
            X_scaled = feature_scaler.fit_transform(X)
            y_scaled = target_scaler.fit_transform(np.array(y).reshape(-1, 1))
            
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            
            # Perform train-test split
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                X_reshaped, y_scaled, X.index, test_size=0.27, random_state=42
            )
            
            # Reshape arrays to 2D for Gaussian Process
            X_train_2D = X_train.reshape(X_train.shape[0], X_train.shape[1])
            X_test_2D = X_test.reshape(X_test.shape[0], X_test.shape[1])
            
            # Apply Gaussian Processes
            gp_pred, gp_sigma = apply_gaussian_processes(X_train_2D, y_train, X_test_2D, y_test)
            gp_rmse = np.sqrt(mean_squared_error(y_test, gp_pred))
            print(f'GP RMSE for {target}: {gp_rmse}')
            
            X_lorenz = df[all_features].values  
            y_lorenz = df[target].values
            
            # Train the Lorenz surrogate model
            lorenz_model, lorenz_scaler = train_lorenz_surrogate_model(X_lorenz, y_lorenz)
            
            # Save the Lorenz surrogate model and its scaler
            lorenz_model.save(f'{data_path}/{df_name}_{target}_lorenz_surrogate_model.h5')
            joblib.dump(lorenz_scaler, f'{data_path}/{df_name}_{target}_lorenz_feature_scaler.pkl')
            
            # Additional Lorenz surrogate model prediction
            X_test_lorenz = df.loc[test_idx, all_features].values
            X_test_lorenz_scaled = lorenz_scaler.transform(X_test_lorenz)
            lorenz_pred = lorenz_model.predict(X_test_lorenz_scaled)

            # Fetch 'd' values for plotting after the train-test split
            d_values = df.loc[test_idx, 'd'].values
            
            # Apply t-SNE
            X_tsne = apply_tsne(X)
            
            lstm_model = Sequential()
            lstm_model.add(LSTM(137, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            lstm_model.add(LSTM(42))
            lstm_model.add(Dense(1))
            
            lstm_model.compile(optimizer=Adadelta(learning_rate=2.718281828459045, rho=0.0729927007, epsilon=6.62607015e-34), loss=loss_option)
            
            lstm_model.fit(X_train, y_train, epochs=137, batch_size=25)
            
            lstm_pred = lstm_model.predict(X_test)
            
            y_test_original = target_scaler.inverse_transform(y_test)
            lstm_pred_original = target_scaler.inverse_transform(lstm_pred)
            
            lstm_rmse = np.sqrt(mean_squared_error(y_test_original, lstm_pred_original))
            print(f'LSTM RMSE for {target}: {lstm_rmse}')
            
            # Modify file names to include df_name (DataFrame name)
            lstm_model.save(f'{data_path}/{df_name}_{target}_lstm_model.h5')
            joblib.dump(feature_scaler, f'{data_path}/{df_name}_{target}_feature_scaler.pkl')
            joblib.dump(target_scaler, f'{data_path}/{df_name}_{target}_target_scaler.pkl')
            
            # Call the plotting functions
            plot_lstm_results(y_test_original.flatten(), lstm_pred_original.flatten(), target, d_values, df_name, visuals_path)
            plot_gp_results(y_test, gp_pred, gp_sigma, target, df_name, visuals_path)
            plot_tsne_results(X_tsne, y, target, df_name, visuals_path)
            
            # Additional code to plot Lorenz results
            plot_lorenz_results(y_test_original.flatten(), lorenz_pred.flatten(), target, d_values, df_name, visuals_path)
