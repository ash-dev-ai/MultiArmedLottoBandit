import time
from fractions import Fraction
from Rewards import mm_groups, pb_groups

# Define the odds for mm_groups
MM_ODDS = {
    'r': Fraction(1, 27),
    '1w+r': Fraction(1, 89),
    '2w+r': Fraction(1, 693),
    '3w': Fraction(1, 606),
    '3w+r': Fraction(1, 14547),
    '4w': Fraction(1, 38792),
    '4w+r': Fraction(1, 931001),
    '5w': Fraction(1, 12607306),
    '5w+r': Fraction(1, 302575350)
}

# Define the odds for pb_groups
PB_ODDS = {
    'r': Fraction.from_float(1 / 38.32),
    '1w+r': Fraction.from_float(1 / 91.98),
    '2w+r': Fraction.from_float(1 / 701.33),
    '3w': Fraction.from_float(1 / 579.76),
    '3w+r': Fraction.from_float(1 / 14494.11),
    '4w': Fraction.from_float(1 / 36525.17),
    '4w+r': Fraction.from_float(1 / 913129.18),
    '5w': Fraction.from_float(1 / 11688053.52),
    '5w+r': Fraction.from_float(1 / 292201338.00)
}

# Record the start time
start_time = time.process_time()

# Create new dictionaries mm_odds and pb_odds and add the odds to them
mm_odds = {}
for group in mm_groups:
    mm_odds[group] = {'odds': MM_ODDS[group]}

pb_odds = {}
for group in pb_groups:
    pb_odds[group] = {'odds': PB_ODDS[group]}

# Calculate and print the elapsed time
end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time:.4f} seconds")