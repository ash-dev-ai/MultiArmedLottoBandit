import numpy as np
from .rule import Rule

class Rule110(Rule):
    def __init__(self, num_range=(69, 26), n_evolutions=10):
        """
        Initialize Rule110 with a specified number range and evolution steps.
        """
        super().__init__(num_range=num_range, n_evolutions=n_evolutions)

    def apply_rule_logic(self, binary_state):
        """
        Applies Rule 110 logic to evolve the binary state.
        
        Rule 110: A cell becomes 1 if:
            - (left == 1 and center == 0) or 
            - (center == 1 and right == 1) or 
            - (center == 1 and right == 0)
        
        :param binary_state: Binary list to transform.
        :return: Transformed binary list according to Rule 110.
        """
        binary_state = np.array(binary_state, dtype=int)
        next_state = np.zeros_like(binary_state)
        for i in range(1, len(binary_state) - 1):  # Avoid boundaries
            left, center, right = binary_state[i - 1], binary_state[i], binary_state[i + 1]
            next_state[i] = int((left == 1 and center == 0) or 
                                (center == 1 and right == 1) or 
                                (center == 1 and right == 0))
        return next_state.tolist()
