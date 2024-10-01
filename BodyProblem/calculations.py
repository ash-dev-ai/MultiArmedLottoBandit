# body/calculations.py
import pandas as pd
import numpy as np

class Body:
    """Represents a body in the system with position, velocity, and acceleration."""
    def __init__(self, name, positions):
        self.name = name
        self.positions = positions
        self.velocity = np.zeros_like(positions)  # Velocity initialized to zero
        self.acceleration = np.zeros_like(positions)  # Acceleration initialized to zero

    def calculate_velocity(self):
        """Calculate velocity as the difference in position over time."""
        self.velocity = np.diff(self.positions, prepend=self.positions[0])

    def calculate_acceleration(self):
        """Calculate acceleration as the difference in velocity over time."""
        self.acceleration = np.diff(self.velocity, prepend=self.velocity[0])

class Calculator:
    """Performs key calculations like velocity, relative distance, gravitational force, and acceleration."""
    G = 6.67430e-11  # Gravitational constant

    def __init__(self, bodies):
        self.bodies = bodies

    def calculate_velocities(self):
        """Calculate velocities for all bodies."""
        for body in self.bodies:
            body.calculate_velocity()

    def calculate_accelerations(self):
        """Calculate accelerations for all bodies."""
        for body in self.bodies:
            body.calculate_acceleration()

    def calculate_relative_distances(self):
        """Calculate the relative distances between each pair of bodies."""
        relative_distances = {}
        for i, body1 in enumerate(self.bodies):
            for j, body2 in enumerate(self.bodies):
                if i < j:
                    relative_distances[f'{body1.name}_{body2.name}'] = np.abs(body1.positions - body2.positions)
        return relative_distances

    def calculate_gravitational_forces(self, relative_distances):
        """Calculate the gravitational forces between each pair of bodies."""
        forces = {}
        for key, dist in relative_distances.items():
            forces[f'force_{key}'] = Calculator.G / (dist**2)
        return forces

class System:
    """Represents the entire system of bodies and performs all calculations."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)
        self.bodies = self._initialize_bodies()

    def _initialize_bodies(self):
        """Initialize the bodies from the dataset."""
        body_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'z']
        bodies = [Body(name, self.data[name].values) for name in body_names]
        return bodies

    def perform_calculations(self):
        """Perform all calculations and update the dataset."""
        calculator = Calculator(self.bodies)

        # Calculate velocities and accelerations
        calculator.calculate_velocities()
        calculator.calculate_accelerations()

        # Update the dataset with velocities and accelerations
        for body in self.bodies:
            self.data[f'vel_{body.name}'] = body.velocity
            self.data[f'acc_{body.name}'] = body.acceleration

        # Calculate relative distances
        relative_distances = calculator.calculate_relative_distances()

        # Update the dataset with relative distances
        for key, dist in relative_distances.items():
            self.data[f'dist_{key}'] = dist

        # Calculate gravitational forces
        forces = calculator.calculate_gravitational_forces(relative_distances)

        # Update the dataset with forces
        for key, force in forces.items():
            self.data[f'force_{key}'] = force

    def save_updated_data(self):
        """Save the updated dataset with calculations."""
        self.data.to_csv(self.filepath, index=False)
        print(f'Calculations added to {self.filepath}')

def main():
    # Apply the system calculations to all datasets
    for variation in ['a', 'b', 'c']:
        filepath = f'./data/prepped_variation_{variation}.csv'
        system = System(filepath)
        system.perform_calculations()
        system.save_updated_data()

if __name__ == '__main__':
    main()


