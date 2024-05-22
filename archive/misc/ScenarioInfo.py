import pandas as pd
from PatternMachine import scenarios

def print_info(name, data):
    print(f"{name} info:")
    print(pd.DataFrame(data).info())
    print(f"\n{name} shape: {data.shape}")
    print(f"\n{name} head:")
    print(pd.DataFrame(data).head())
    print(f"\n{name} tail:")
    print(pd.DataFrame(data).tail())
    print("\n")

for scenario_name, X, y in scenarios:
    print(f"{scenario_name}:")
    print_info("X", X)
    print_info("y", y)