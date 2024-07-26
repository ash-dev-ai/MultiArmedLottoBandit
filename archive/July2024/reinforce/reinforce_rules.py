# reinforce_rules.py

import numpy as np
import pandas as pd

class ReinforceRules:
    def __init__(self, number_type):
        self.number_type = number_type

    # Existing rules
    def validate_odd_even_rule(self, prediction):
        num1_to_num5 = np.array([prediction['num1'], prediction['num2'], prediction['num3'], prediction['num4'], prediction['num5']])
        odds = np.sum(num1_to_num5 % 2 != 0)
        evens = np.sum(num1_to_num5 % 2 == 0)
        return odds in [2, 3] and evens in [2, 3]

    def validate_similarity_rule(self, prediction, historical_data):
        num1_to_num5 = np.array([prediction['num1'], prediction['num2'], prediction['num3'], prediction['num4'], prediction['num5']])
        historical_matches = historical_data.apply(lambda row: np.sum(row[:5] == num1_to_num5), axis=1)
        return not (historical_matches >= 3).any()

    def validate_unique_numbers_rule(self, prediction):
        num1_to_num5 = [prediction['num1'], prediction['num2'], prediction['num3'], prediction['num4'], prediction['num5']]
        return len(set(num1_to_num5)) == len(num1_to_num5)

    def validate_num_sum_rule(self, prediction):
        num1_to_num5 = np.array([prediction['num1'], prediction['num2'], prediction['num3'], prediction['num4'], prediction['num5']])
        numSum = num1_to_num5.sum()
        return 140 <= numSum <= 240

    def validate_probabilities_and_ratios_rule(self, prediction):
        num1_to_num5 = np.array([prediction['num1'], prediction['num2'], prediction['num3'], prediction['num4'], prediction['num5']])
        odds = np.sum(num1_to_num5 % 2 != 0)
        evens = np.sum(num1_to_num5 % 2 == 0)

        if self.number_type['num1-num5'] == 69:  # Powerball
            total_combinations = 292201338
            balanced_comb = 1846389480
            p_balanced = balanced_comb / total_combinations
        elif self.number_type['num1-num5'] == 70:  # Mega Millions
            total_combinations = 302575350
            balanced_comb = 2034382500
            p_balanced = balanced_comb / total_combinations
        else:  # Combined
            pb_total_combinations = 292201338
            pb_balanced_comb = 1846389480
            pb_p_balanced = pb_balanced_comb / pb_total_combinations

            mm_total_combinations = 302575350
            mm_balanced_comb = 2034382500
            mm_p_balanced = mm_balanced_comb / mm_total_combinations

            p_balanced = (pb_p_balanced + mm_p_balanced) / 2

        if (odds == 3 and evens == 2) or (evens == 3 and odds == 2):
            return np.random.rand() <= p_balanced

        return True

    # New rules
    def frequency_based_rule(self, prediction, historical_data):
        # Identify the most frequently occurring numbers in historical data
        all_numbers = historical_data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].values.flatten()
        frequency_counts = pd.Series(all_numbers).value_counts()
        most_frequent_numbers = frequency_counts.index[:6].tolist()
        return any(num in most_frequent_numbers for num in prediction.values())

    def hot_and_cold_numbers_rule(self, prediction, historical_data):
        # Separate numbers into "hot" (frequently drawn) and "cold" (rarely drawn) groups
        all_numbers = historical_data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].values.flatten()
        frequency_counts = pd.Series(all_numbers).value_counts()
        hot_numbers = frequency_counts.index[:6].tolist()
        cold_numbers = frequency_counts.index[-6:].tolist()
        return any(num in hot_numbers for num in prediction.values())

    def consecutive_number_rule(self, prediction):
        # Check if consecutive numbers are often present in the prediction
        numbers = sorted(prediction.values())
        consecutive_count = sum(1 for i in range(len(numbers) - 1) if numbers[i] + 1 == numbers[i + 1])
        return consecutive_count >= 2

    def odd_even_number_rule(self, prediction):
        # Analyze the distribution of odd and even numbers in the prediction
        odd_count = sum(1 for num in prediction.values() if num % 2 != 0)
        even_count = sum(1 for num in prediction.values() if num % 2 == 0)
        return odd_count >= 2 and even_count >= 2

    def number_sum_rule(self, prediction, historical_data):
        # Calculate the sum of the numbers in the prediction and check against historical data
        prediction_sum = sum(prediction.values())
        historical_sums = historical_data[['num1', 'num2', 'num3', 'num4', 'num5']].sum(axis=1)
        sum_range = historical_sums.quantile([0.25, 0.75]).values
        return sum_range[0] <= prediction_sum <= sum_range[1]

    def number_patterns_and_sequences_rule(self, prediction, historical_data):
        # Search for recurring number patterns or sequences in the prediction
        pattern_found = False
        for i in range(len(historical_data) - 5):
            if set(prediction.values()) == set(historical_data.iloc[i:i+5][['num1', 'num2', 'num3', 'num4', 'num5']].values.flatten()):
                pattern_found = True
                break
        return pattern_found

    def delayed_numbers_rule(self, prediction, historical_data):
        # Check for delayed numbers in the prediction
        last_appearance = historical_data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].apply(lambda x: x == prediction.values(), axis=1).idxmax()
        return (historical_data.index[-1] - last_appearance).days >= 30

    def historical_performance_rule(self, prediction, historical_data):
        # Evaluate the performance of the numbers in the prediction based on historical data
        all_numbers = historical_data[['num1', 'num2', 'num3', 'num4', 'num5', 'numA']].values.flatten()
        frequency_counts = pd.Series(all_numbers).value_counts()
        historical_performance = sum(frequency_counts[num] for num in prediction.values() if num in frequency_counts)
        return historical_performance >= frequency_counts.mean()

    def random_sampling_rule(self, prediction):
        # In the absence of strong patterns, randomly sample from the pool of numbers
        all_numbers = list(range(1, 70))  # Assuming 1 to 69 for num1 to num5, and 1 to 26 for numA
        sampled_numbers = np.random.choice(all_numbers, size=6, replace=False)
        return any(num in sampled_numbers for num in prediction.values())

    # Method to validate all rules
    def validate_all_rules(self, prediction, historical_data):
        return (self.validate_odd_even_rule(prediction) and
                self.validate_similarity_rule(prediction, historical_data) and
                self.validate_unique_numbers_rule(prediction) and
                self.validate_num_sum_rule(prediction) and
                self.validate_probabilities_and_ratios_rule(prediction) and
                self.frequency_based_rule(prediction, historical_data) and
                self.hot_and_cold_numbers_rule(prediction, historical_data) and
                self.consecutive_number_rule(prediction) and
                self.odd_even_number_rule(prediction) and
                self.number_sum_rule(prediction, historical_data) and
                self.number_patterns_and_sequences_rule(prediction, historical_data) and
                self.delayed_numbers_rule(prediction, historical_data) and
                self.historical_performance_rule(prediction, historical_data) and
                self.random_sampling_rule(prediction))

