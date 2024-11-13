import numpy as np
from .rule import Rule

class Rule150(Rule):
    def __init__(self, num_range=(69, 26), n_evolutions=10):
        """
        Initialize Rule150 with specified number ranges and evolution steps.
        """
        super().__init__(num_range=num_range, n_evolutions=n_evolutions)

    def apply_rule_logic(self, binary_state):
        """
        Applies Rule 150 logic to evolve the binary state.
        
        Rule 150: Each cell evolves by taking the XOR of its left, center, and right neighbors.
        
        :param binary_state: List of binary values to evolve.
        :return: The next state as per Rule 150.
        """
        binary_state = np.array(binary_state, dtype=int)
        next_state = np.zeros_like(binary_state)
        for i in range(1, len(binary_state) - 1):  # Avoid boundaries
            left, center, right = binary_state[i - 1], binary_state[i], binary_state[i + 1]
            next_state[i] = left ^ center ^ right  # XOR of neighbors
        return next_state.tolist()
