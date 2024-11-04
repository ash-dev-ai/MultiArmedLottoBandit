import numpy as np
import pandas as pd

class Rule90:
    def __init__(self, num_range=(69, 26)):
        self.num_range_main, self.num_range_A = num_range
    
    def apply_rule_90(self, binary_state):
        """
        Applies Rule 90 to evolve the binary state.
        Rule 90: If left and right neighbors are the same, set the center to 1; otherwise, set it to 0.
        """
        binary_state = binary_state.astype(int)
        next_state = np.zeros_like(binary_state)
        for i in range(1, len(binary_state) - 1):  # Avoid boundaries
            left, center, right = binary_state[i - 1], binary_state[i], binary_state[i + 1]
            next_state[i] = int((left == 1 and right == 1) or (left == 0 and right == 0))
        return next_state
    
    def convert_to_binary(self, row):
        """
        Convert the lottery numbers into binary state based on num range.
        """
        binary_state = np.zeros(self.num_range_main)
        for num in [row['num1'], row['num2'], row['num3'], row['num4'], row['num5']]:
            binary_state[num - 1] = 1  # Adjust for 0-indexing

        binary_stateA = np.zeros(self.num_range_A)
        if 1 <= row['numA'] <= self.num_range_A:
            binary_stateA[row['numA'] - 1] = 1  # Adjust for 0-indexing

        return binary_state, binary_stateA

    def apply_rule_to_prediction(self, model_predictions):
        """
        Applies Rule 90 transformations to adjust model-based predictions.
        
        :param model_predictions: Initial predictions from models.
        :return: Adjusted predictions.
        """
        binary_state = np.zeros(self.num_range_main)
        for num in model_predictions['num1-5']:
            binary_state[num - 1] = 1

        binary_stateA = np.zeros(self.num_range_A)
        if 1 <= model_predictions['numA'] <= self.num_range_A:
            binary_stateA[model_predictions['numA'] - 1] = 1

        # Evolve the binary states
        binary_state = self.apply_rule_90(binary_state)
        binary_stateA = self.apply_rule_90(binary_stateA)

        # Convert binary states back to lottery numbers
        adjusted_nums = np.where(binary_state == 1)[0] + 1  # Adjust for 1-based indexing
        adjusted_numA = np.where(binary_stateA == 1)[0] + 1  # Adjust for 1-based indexing

        # Ensure exactly 5 numbers for num1-5 and one for numA
        if len(adjusted_nums) >= 5:
            adjusted_nums = np.random.choice(adjusted_nums, 5, replace=False)
        else:
            remaining_nums = np.setdiff1d(np.arange(1, self.num_range_main + 1), adjusted_nums)
            adjusted_nums = np.concatenate([adjusted_nums, np.random.choice(remaining_nums, 5 - len(adjusted_nums), replace=False)])

        if len(adjusted_numA) > 0:
            adjusted_numA = np.random.choice(adjusted_numA, 1)[0]
        else:
            adjusted_numA = np.random.choice(np.arange(1, self.num_range_A + 1), 1)[0]

        return {
            "num1-5": np.sort(adjusted_nums),
            "numA": adjusted_numA
        }

    def generate_predictions(self, data, n_predictions=3, n_past_draws=5, n_evolutions=3):
        """
        Generates n predictions using Rule 90 based on the last n_past_draws.
        """
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
                binary_states = self.apply_rule_90(binary_states)
                binary_statesA = self.apply_rule_90(binary_statesA)

            predicted_nums = np.where(binary_states == 1)[0] + 1
            predicted_numA = np.where(binary_statesA == 1)[0] + 1

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