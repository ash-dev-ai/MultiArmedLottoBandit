import cmath

def quantum_interference(num):
    """Simulate a quantum interference operation on a given number."""
    theta = cmath.tau / 8  # Choose an angle (in radians), using tau instead of pi
    return num * cmath.exp(1j * theta)  # Apply the interference operation

# Initialize the list of numbers
numbers = [6, 7, 8, 32, 44, 13]

# Apply quantum interference to each number
interfered_numbers = [quantum_interference(num) for num in numbers]

# Display the results
print("Original numbers:", numbers)
print("Numbers after interference:", interfered_numbers)
