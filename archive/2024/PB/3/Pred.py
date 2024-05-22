# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:52:09 2024

@author: AVMal
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import itertools

pb_stats = {
    'Mean': 15.498110831234257, 
    'Median': 15.0, 
    'Mode': 24, 
    'Standard Deviation': 9.166079296040747
}

winning_numbers_sum_stats = {
    'Mean': 166.9294710327456, 
    'Median': 168.5, 
    'Mode': 158, 
    'Standard Deviation': 44.0767511038843
}

def load_latest_dataset():
    base_path = 'data/prep'
    latest_date = sorted(os.listdir(base_path), reverse=True)[0]  # Get the latest date
    latest_time = sorted(os.listdir(f"{base_path}/{latest_date}"), reverse=True)[0]  # Get the latest time for that date
    dataset_path = f"{base_path}/{latest_date}/{latest_time}/org_df.csv"
    return pd.read_csv(dataset_path)

def check_odd_even_distribution(numbers):
    odd_count = sum(1 for num in numbers if num % 2 != 0)
    even_count = len(numbers) - odd_count
    return (odd_count == 3 and even_count == 2) or (odd_count == 2 and even_count == 3)

def is_similar_to_recent_draws(numbers, recent_draws, threshold=3):
    for draw in recent_draws:
        match_count = sum(num in draw for num in numbers)
        if match_count >= threshold:
            return True 
    return False

def is_within_w_range(numbers):
    """Check if white ball numbers are within the range 1-70."""
    return all(1 <= num <= 69 for num in numbers[:-1])

def is_within_pb_range(number):
    """Check if the Powerball number is within the range 1-26."""
    return 1 <= number <= 26

def check_number_groups(numbers):
    """Ensure numbers fall into the correct number groups with 3 or 4-number groups only."""
    groups = [0] * 7  # For each decade group: 1-9, 10-19, 20-29, ..., 60-69
    for num in numbers:  # Include all white balls, exclude PB for group count
        group_index = (num - 1) // 10  # Calculate group index based on number
        groups[group_index] += 1

    non_empty_groups = sum(1 for count in groups if count > 0)
    return 3 <= non_empty_groups <= 4


def check_consecutive_pairs(numbers):
    """Allow only pairs of consecutive numbers."""
    sorted_nums = sorted(numbers[:-1])  # Exclude Powerball
    consecutive_pairs = sum(1 for i in range(len(sorted_nums) - 1) if sorted_nums[i + 1] - sorted_nums[i] == 1)
    return consecutive_pairs == 1

def check_winning_numbers_sum(numbers):
    """Check if the sum of the numbers is between 140 and 240."""
    return 124 <= sum(numbers[:-1]) <= 212  # Exclude Powerball

def generate_valid_predictions(model_w, model_pb, X_features, recent_draws):
    valid_sets = []
    attempts = 0

    while len(valid_sets) < 5 and attempts < 1000:
        # Generate predictions
        w_predictions = model_w.predict(X_features).tolist()
        pb_prediction = model_pb.predict(X_features)[0]
        full_prediction = w_predictions + [pb_prediction]

        # Perform all validation checks
        if (is_within_w_range(full_prediction) and
            is_within_pb_range(pb_prediction) and
            check_odd_even_distribution(full_prediction[:-1]) and
            check_number_groups(full_prediction[:-1]) and
            check_consecutive_pairs(full_prediction[:-1]) and
            check_winning_numbers_sum(full_prediction[:-1]) and
            not is_similar_to_recent_draws(full_prediction[:-1], recent_draws)):
            valid_sets.append((sorted(w_predictions), pb_prediction))

        attempts += 1

    return valid_sets

def validate_predictions(predictions, target):
    """
    Ensure predictions are within the valid range.
    """
    if target == 'pb':
        # Ensure Powerball predictions are within 1-26
        return [p if 1 <= p <= 26 else 26 for p in predictions]
    else:
        # Ensure white ball predictions are within 1-70
        return [p if 1 <= p <= 69 else 69 for p in predictions]

def main():
    df = load_latest_dataset()

    # Basic features
    features = ['winning_numbers_sum', 'd']
    X_basic = df[features]

    # Creating new features based on statistics for variability
    df['sum_deviation_from_mean'] = abs(df['winning_numbers_sum'] - winning_numbers_sum_stats['Mean'])
    df['pb_deviation_from_median'] = abs(df['pb'] - pb_stats['Median'])

    # Extended feature set including statistical features
    features_extended = features + ['sum_deviation_from_mean', 'pb_deviation_from_median']
    X_extended = df[features_extended]

    targets = ['w1', 'w2', 'w3', 'w4', 'w5', 'pb']
    predictions_sets = []  # List to store aggregated predictions for each set

    # Initialize a dictionary to store individual predictions for aggregation
    individual_predictions = {target: [] for target in targets}

    for target in targets:
        print(f"Predicting {target} using extended features for added variability...")
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X_extended, y, test_size=0.20, random_state=None)

        model = RandomForestClassifier(random_state=None)
        model.fit(X_train, y_train)

        raw_predictions = model.predict(X_test)
        validated_predictions = validate_predictions(raw_predictions, target)

        # Store the validated predictions for aggregation
        individual_predictions[target] = validated_predictions

        print(f"Sample validated predictions for {target}: {validated_predictions[:5]}")

    # After collecting all predictions, print them as sets
    for i in range(len(individual_predictions['w1'])):  # Assuming all targets have the same number of predictions
        prediction_set = tuple(individual_predictions[target][i] for target in ['w1', 'w2', 'w3', 'w4', 'w5', 'pb'])
        predictions_sets.append(prediction_set)

    # Print the aggregated set views
    for i, prediction_set in enumerate(predictions_sets[:5], start=1):  # Limiting output for brevity
        print(f"Prediction set {i}: {prediction_set}")

if __name__ == "__main__":
    main()
