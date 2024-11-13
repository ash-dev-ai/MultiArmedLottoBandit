import numpy as np
from .rule import Rule

class Rule90(Rule):
    def __init__(self, num_range=(69, 26), n_evolutions=10):
        """
        Initialize Rule90 with a specified number range and evolution steps.
        """
        super().__init__(num_range=num_range, n_evolutions=n_evolutions)

    def apply_rule_logic(self, binary_state):
        """
        Applies Rule 90 logic to evolve the binary state.
        
        Rule 90: If left and right neighbors are the same, set the center to 1; otherwise, set it to 0.
        
        :param binary_state: Binary list to transform.
        :return: Transformed binary list according to Rule 90.
        """
        binary_state = np.array(binary_state, dtype=int)
        next_state = np.zeros_like(binary_state)
        for i in range(1, len(binary_state) - 1):  # Avoid boundaries
            left, center, right = binary_state[i - 1], binary_state[i], binary_state[i + 1]
            next_state[i] = int((left == right))  # Center becomes 1 if left == right
        return next_state.tolist()
