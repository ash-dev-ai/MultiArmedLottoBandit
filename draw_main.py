# Import necessary modules
import os
import pandas as pd
from draw import LotteryVisualizer

def main():
    # List of dataset names and their max numbers for num1-num5 and numA
    datasets = [
        {"name": "data_combined.csv", "max_num": 70, "max_numA": 26, "is_comb": True},
        {"name": "data_pb.csv", "max_num": 69, "max_numA": 26, "is_pb": True},
        {"name": "data_mb.csv", "max_num": 70, "max_numA": 25}
    ]

    # List of rule numbers to apply
    rule_numbers = [30, 37, 42, 45, 73, 90, 110, 150, 254]

    # Iterate over each dataset
    for dataset_info in datasets:
        dataset_name = dataset_info["name"]
        max_num = dataset_info["max_num"]
        max_numA = dataset_info["max_numA"]
        is_pb = dataset_info.get("is_pb", False)
        is_comb = dataset_info.get("is_comb", False)
        
        # Initialize the visualizer for the current dataset
        visualizer = LotteryVisualizer(
            file_path=os.path.join("D:/Projects/Project-PBMM/current/MultiArmedLottoBandit/data", dataset_name),
            max_num=max_num,
            max_numA=max_numA,
            is_pb=is_pb,
            is_comb=is_comb
        )

        # Create cumulative images for the dataset
        output_dir_cumulative = f"images/{dataset_name.split('.')[0]}_images"
        visualizer.create_cumulative_image(output_dir=output_dir_cumulative, filename=f"{dataset_name.split('.')[0]}_cumulative_image", limit=600)

        # Iterate over each rule number for automata images
        for rule_number in rule_numbers:
            output_dir_automata = f"images/{dataset_name.split('.')[0]}_automata"
            visualizer.create_automata_image(rule_number=rule_number, output_dir=output_dir_automata, filename=f"{dataset_name.split('.')[0]}_automata_image_rule_{rule_number}", limit=100)

if __name__ == "__main__":
    main()
