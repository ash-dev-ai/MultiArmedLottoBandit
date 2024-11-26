from gravity.pickMachine import GravityPickMachine

def main():
    # Initialize Mega Millions machine
    mega_millions_machine = GravityPickMachine(
        num_balls=70,     # White balls (1–70)
        num_selected=5,   # Number of white balls to pick
        mega_balls=25     # Mega Ball (1–25)
    )

    # Initialize Powerball machine
    powerball_machine = GravityPickMachine(
        num_balls=69,     # White balls (1–69)
        num_selected=5,   # Number of white balls to pick
        mega_balls=26     # Powerball (1–26)
    )

    # Run simulations
    print("Running Mega Millions simulation...")
    mega_results = mega_millions_machine.run_simulation(
        drum_speed=70,     # RPM for drum speed
        air_jet_force=26,   # Air jet force scale
        mixing_time=30     # Mixing time in seconds
    )

    print("Running Powerball simulation...")
    powerball_results = powerball_machine.run_simulation(
        drum_speed=70,     # RPM for drum speed
        air_jet_force=26,   # Air jet force scale
        mixing_time=30      # Mixing time in seconds
    )

    # Display results
    print("\nResults:")
    print("Mega Millions:", mega_results)
    print("Powerball:", powerball_results)

    # Save results to files
    mega_millions_machine.save_results("data/mega_millions_results.csv", mega_results)
    powerball_machine.save_results("data/powerball_results.csv", powerball_results)

    print("\nResults saved to:")
    print(" - data/mega_millions_results.csv")
    print(" - data/powerball_results.csv")


if __name__ == "__main__":
    main()
