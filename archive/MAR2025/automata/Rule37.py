import numpy as np
from .rule import Rule

class Rule37(Rule):
    def __init__(self, num_range=(69, 26), n_evolutions=10):
        """
        Initialize Rule37 with a specified number range and evolution steps.
        """
        super().__init__(num_range=num_range, n_evolutions=n_evolutions)

    def apply_rule_logic(self, binary_state):
        """
        Applies Rule 37 logic to evolve the binary state.
        
        Rule 37: (left == 0 and center == 1) or (center == 1 and right == 0) or (left == 1 and right == 1).
        
        :param binary_state: Binary list to transform.
        :return: Transformed binary list according to Rule 37.
        """
        binary_state = np.array(binary_state, dtype=int)
        next_state = np.zeros_like(binary_state)
        for i in range(1, len(binary_state) - 1):  # Avoid boundaries
            left, center, right = binary_state[i - 1], binary_state[i], binary_state[i + 1]
            next_state[i] = int((left == 0 and center == 1) or (center == 1 and right == 0) or (left == 1 and right == 1))
        return next_state.tolist()
