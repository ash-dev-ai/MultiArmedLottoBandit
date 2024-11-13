import random

class Rule:
    def __init__(self, num_range, n_evolutions=10):
        self.num_range = num_range
        self.n_evolutions = n_evolutions

    def apply(self, base_prediction):
        binary_state = self.convert_to_binary(base_prediction)
        evolved_state = self.evolve(binary_state)
        adjusted_prediction = self.apply_to_prediction(evolved_state)
        return self.generate_prediction(adjusted_prediction)

    def convert_to_binary(self, prediction):
        binary_state = []
        for num in prediction['num1-5'] + [prediction['numA']]:
            binary_state.extend([int(x) for x in bin(num)[2:].zfill(8)])
        return binary_state

    def evolve(self, binary_state):
        for _ in range(self.n_evolutions):
            binary_state = self.apply_rule_logic(binary_state)
        return binary_state

    def apply_rule_logic(self, binary_state):
        raise NotImplementedError("Each rule must implement its own `apply_rule_logic` method.")

    def apply_to_prediction(self, evolved_state):
        # Initialize unique numbers with a partitioned sampling
        main_numbers = set()
        partitioned_range = [
            (1, self.num_range[0] // 3),
            (self.num_range[0] // 3, 2 * self.num_range[0] // 3),
            (2 * self.num_range[0] // 3, self.num_range[0])
        ]
        for low, high in partitioned_range:
            if len(main_numbers) < 5:
                main_numbers.add(random.randint(low, high))

        # Ensure exactly five unique numbers in main_numbers
        if len(main_numbers) < 5:
            remaining_numbers = sorted(set(range(1, self.num_range[0] + 1)) - main_numbers)
            main_numbers.update(random.sample(remaining_numbers, 5 - len(main_numbers)))

        main_numbers = sorted(list(main_numbers))[:5]  # Final unique numbers

        # Adaptive selection for `numA`
        numA = evolved_state[5] if len(evolved_state) > 5 else random.randint(1, self.num_range[1])
        if numA >= self.num_range[1] - 1:
            numA = random.randint(1, self.num_range[1] // 2)
        elif numA <= 2:
            numA = random.randint(self.num_range[1] // 2, self.num_range[1])
        else:
            numA += random.choice([-2, 2])

        numA = max(1, min(numA, self.num_range[1]))  # Clamp within range

        return main_numbers, numA

    def generate_prediction(self, adjusted):
        nums, numA = adjusted
        return {
            'num1-5': nums,
            'numA': numA
        }

    def generate_predictions(self, data, n_predictions=1):
        predictions = []
        for _ in range(n_predictions):
            base_prediction = {
                'num1-5': random.sample(range(1, self.num_range[0] + 1), 5),
                'numA': random.randint(1, self.num_range[1])
            }
            prediction = self.apply(base_prediction)
            predictions.append(prediction)
        return predictions
