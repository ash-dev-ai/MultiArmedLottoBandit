import numpy as np
import pandas as pd
import pickle
import json
import ast
from scipy.stats import shapiro
from scipy.fft import fft
from scipy.stats import probplot
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt

# Custom JSON encoder for numpy arrays and complex numbers
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        return json.JSONEncoder.default(self, obj)

def plot_results_from_df(df):
    for index, row in df.iterrows():
        label = row['Label']
        column = row['Column']
        results_data_list = row['Results Data']
        
        print(f"Debug Info: Label={label}, Column={column}, Type of Results Data List={type(results_data_list)}")
        
        if isinstance(results_data_list, list):
            for results_data in results_data_list:
                print(f"Debug Info: Type of Results Data={type(results_data)}")
                
                if isinstance(results_data, dict):
                    try:
                        shapiro_statistic = results_data.get('Shapiro-Wilk_test_statistic', None)
                        fourier_terms = results_data.get('Fourier_terms', None)
                        bayesian_mean = results_data.get('Bayesian_mean', None)
                        
                        if shapiro_statistic is not None:
                            print(f"Shapiro-Wilk Test Statistic for {label}, Column: {column}: {shapiro_statistic}")
                            plt.figure()
                            plt.hist(shapiro_statistic, bins=20)
                            plt.title(f"Histogram of Shapiro-Wilk Test Statistic for {label}, Column: {column}")
                            plt.show()
                        
                        print(f"Debug Info: Type of Fourier Terms={type(fourier_terms)}")
                        
                        if fourier_terms is not None and isinstance(fourier_terms, np.ndarray):
                            plt.figure()
                            plt.plot(fourier_terms)
                            plt.title(f"Fourier Terms for {label}, Column: {column}")
                            plt.show()
                            
                        print(f"Debug Info: Type of Bayesian Mean={type(bayesian_mean)}")
                        
                        if bayesian_mean is not None and isinstance(bayesian_mean, np.ndarray):
                            plt.figure()
                            plt.hist(bayesian_mean, bins=20)
                            plt.title(f"Histogram of Bayesian Mean for {label}, Column: {column}")
                            plt.show()
                            
                            plt.figure()
                            autocorrelation_plot(bayesian_mean)
                            plt.title(f"Autocorrelation Plot for {label}, Column: {column}")
                            plt.show()

                            plt.figure()
                            probplot(bayesian_mean, plot=plt)
                            plt.title(f"Q-Q Plot for {label}, Column: {column}")
                            plt.show()
                            
                    except Exception as e:
                        print(f"Could not process data for {label}, Column: {column}. Error: {e}")
                else:
                    print(f"Skipping unsupported data type in list for {label}, Column: {column}.")
        else:
            print(f"Skipping unsupported data type for {label}, Column: {column}.")


#Data Processing
try:
    with open('checkpoint.pkl', 'rb') as f:
        all_fits, all_results = pickle.load(f)
except FileNotFoundError:
    print("Error: The 'checkpoint.pkl' file was not found.")
    all_fits, all_results = {}, {}

# Create an empty list to store DataFrame rows
df_rows = []

# Populate the DataFrame
for label, columns_data in all_results.items():
    for column, results_data in columns_data.items():
        row_dict = {'Label': label, 'Column': column, 'Results Data': results_data}
        df_rows.append(row_dict)

# Create DataFrame
df = pd.DataFrame(df_rows)

# Convert DataFrame to dictionary
df_dict = df.to_dict(orient='records')

# Serialize dictionary to JSON
df_json = json.dumps(df_dict, cls=NumpyEncoder)

# Save JSON data to file
with open('results_dataframe.json', 'w') as json_file:
    json_file.write(df_json)

# Plot results
plot_results_from_df(df)