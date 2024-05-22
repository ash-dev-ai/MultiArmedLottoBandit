# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 01:07:40 2023

@author: AVMal
"""

import pickle
import json
import numpy as np

# Global variables with some sample data
all_fits = {
    'parameter1': [1.0, 2.0, 3.0],
    'parameter2': [4.0, 5.0, 6.0]
}
all_results = {
    'result1': [10.0, 20.0, 30.0],
    'result2': [40.0, 50.0, 60.0]
}

# Define a custom JSON encoder to handle NumPy ndarrays and complex numbers
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}  # Convert complex to dictionary
        return super(NumpyEncoder, self).default(obj)

# Custom JSON Decoder
def decode_complex(dct):
    if 'real' in dct and 'imag' in dct:
        return complex(dct['real'], dct['imag'])
    return dct

def save_checkpoint(checkpoint_file="./data/study0_ckpnt.pkl"):
    global all_fits, all_results
    print("Saving checkpoint...")
    # Debugging: Check data before saving
    print(f"Data before saving: all_fits={all_fits}, all_results={all_results}")
    try:
        with open(checkpoint_file, "wb") as f:
            pickle.dump((all_fits, all_results), f)
        print("Checkpoint saved.")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

# Call the save_checkpoint function to create the initial pickle file
save_checkpoint()