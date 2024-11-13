import numpy as np
from .rule import Rule

class Rule254(Rule):
    def __init__(self, num_range=(69, 26), n_evolutions=10):
        """
        Initialize Rule254 with specified number ranges and evolution steps.
        """
        super().__init__(num_range=num_range, n_evolutions=n_evolutions)

    def apply_rule_logic(self, binary_state):
        """
        Applies Rule 254 logic to evolve the binary state.

        Rule 254: Sets the next state to 1 if any of the left, center, or right bits are 1.

        :param binary_state: List of binary values to evolve.
        :return: The next state as per Rule 254.
        """
        binary_state = np.array(binary_state, dtype=int)
        next_state = np.zeros_like(binary_state)
        for i in range(1, len(binary_state) - 1):  # Avoid boundaries
            left, center, right = binary_state[i - 1], binary_state[i], binary_state[i + 1]
            next_state[i] = int((left == 1) or (center == 1) or (right == 1))
        return next_state.tolist()
