# Rule150.py
import numpy as np
import pandas as pd

class Rule150:
    def __init__(self, num_range=(69, 26)):
        self.num_range_main, self.num_range_A = num_range
    
    def apply_rule_150(self, binary_state):
        """
        Applies Rule 150 to evolve the binary state.
        Rule 150: XOR of left, center, and right neighbors.
        """
        binary_state = binary_state.astype(int)
        next_state = np.zeros_like(binary_state)
        for i in range(1, len(binary_state) - 1):  # Avoid boundaries
            left, center, right = binary_state[i - 1], binary_state[i], binary_state[i + 1]
            next_state[i] = left ^ center ^ right
        return next_state
    
    def convert_to_binary(self, row):
        """
        Convert the lottery numbers into binary state based on num range.
        Each number is represented by setting its index in the array to 1.
        """
        # Binary for num1 to num5 (range: 1-num_range_main)
        binary_state = np.zeros(self.num_range_main)
        for num in [row['num1'], row['num2'], row['num3'], row['num4'], row['num5']]:
            binary_state[num - 1] = 1  # Adjust for 0-indexing

        # Binary for numA (range: 1-num_range_A)
        binary_stateA = np.zeros(self.num_range_A)
        if 1 <= row['numA'] <= self.num_range_A:
            binary_stateA[row['numA'] - 1] = 1  # Adjust for 0-indexing

        return binary_state, binary_stateA

    def generate_predictions(self, data, n_predictions=3, n_past_draws=5, n_evolutions=3):
        """
        Generates n predictions using Rule 150 based on the last n_past_draws.
        For each prediction, evolve the state n_evolutions times.
        """
        # Get the last n_past_draws to initialize the binary state
        past_draws = data.iloc[-n_past_draws:]
        binary_states = np.zeros(self.num_range_main)
        binary_statesA = np.zeros(self.num_range_A)

        # Accumulate the binary states from past draws
        for _, row in past_draws.iterrows():
            binary_state, binary_stateA = self.convert_to_binary(row)
            binary_states += binary_state
            binary_statesA += binary_stateA

        # Normalize the accumulated binary state (turn into binary 0/1)
        binary_states = np.clip(binary_states, 0, 1)
        binary_statesA = np.clip(binary_statesA, 0, 1)

        predictions = []

        for _ in range(n_predictions):
            # Evolve the binary state multiple times
            for _ in range(n_evolutions):
                binary_states = self.apply_rule_150(binary_states)
                binary_statesA = self.apply_rule_150(binary_statesA)

            # Convert the evolved states back to lottery numbers
            predicted_nums = np.where(binary_states == 1)[0] + 1  # Adjust for 1-based indexing
            predicted_numA = np.where(binary_statesA == 1)[0] + 1  # Adjust for 1-based indexing

            # Get the top 5 numbers and one Powerball-like number
            if len(predicted_nums) >= 5:
                predicted_nums = np.random.choice(predicted_nums, 5, replace=False)
            else:
                remaining_nums = np.setdiff1d(np.arange(1, self.num_range_main + 1), predicted_nums)
                predicted_nums = np.concatenate([predicted_nums, np.random.choice(remaining_nums, 5 - len(predicted_nums), replace=False)])

            if len(predicted_numA) > 0:
                predicted_numA = np.random.choice(predicted_numA, 1)[0]
            else:
                predicted_numA = np.random.choice(np.arange(1, self.num_range_A + 1), 1)[0]

            predictions.append({
                "num1-5": np.sort(predicted_nums),
                "numA": predicted_numA
            })

        return predictions

# Example of how to use the class
if __name__ == "__main__":
    # Load the dataset (adjust the path as needed)
    dataset_path = 'data/data_combined.csv'
    data = pd.read_csv(dataset_path)
    
    # Instantiate Rule150 class with the appropriate number range
    rule_150 = Rule150(num_range=(70, 26))
    
    # Generate 3 predictions
    predictions = rule_150.generate_predictions(data, n_predictions=3)
    
    # Output the predictions
    for i, prediction in enumerate(predictions, start=1):
        print(f"Prediction {i}: num1-5 = {prediction['num1-5']}, numA = {prediction['numA']}")
