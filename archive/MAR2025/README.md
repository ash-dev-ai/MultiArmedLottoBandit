### README.md

# MultiArmedLottoBandit

This project processes and analyzes Powerball and Mega Millions lottery data to identify patterns and trends using various statistical and analytical methods.

## Directory Structure

```
/
prep_main.py
- prep/
  - num_main.py
  - <prep files>
  - num/
    - prep_num_pb.py
    - prep_num_mb.py
    - count.py  
    - <other num scripts> # Scripts for other analyses such as frequency, hot/cold numbers, etc.
data/
  - data_pb.csv
  - data_mb.csv
  - data_combined.csv
  - num_pb.csv
  - num_mb.csv
  - num_combined.csv
```

## Files and Scripts

### prep_main.py
The main script that orchestrates the entire data preparation and analysis workflow. It calls various preprocessing scripts and ensures the processed data is ready for analysis.

### prep/
This directory contains all preprocessing files and scripts necessary for preparing the Powerball and Mega Millions data.

#### num_main.py
Calls the number preparation scripts for both Powerball and Mega Millions.

#### num/
Contains scripts for various analyses related to the lottery numbers.

##### prep_num_pb.py
Loads Powerball data, counts the occurrences of each number, and saves the results.

##### prep_num_mb.py
Loads Mega Millions data, counts the occurrences of each number, and saves the results.

##### count.py
Handles the counting of occurrences of each number for both Powerball and Mega Millions. This script is used by `prep_num_pb.py` and `prep_num_mb.py`.

##### <other num scripts>
Scripts for other analyses such as frequency, hot/cold numbers, etc.

### data/
Contains the raw and processed data files.

- **data_pb.csv**: Raw Powerball data.
- **data_mb.csv**: Raw Mega Millions data.
- **data_combined.csv**: Combined dataset of Powerball and Mega Millions data.
- **num_pb.csv**: Processed Powerball data with number counts.
- **num_mb.csv**: Processed Mega Millions data with number counts.
- **num_combined.csv**: Processed combined data with number counts (to be implemented).

## Setup and Usage

### Prerequisites
- Python 3.x
- pandas library

### Installing Dependencies
Install the required Python packages:
```bash
pip install pandas
```

### Running the Project
1. Ensure the raw data files (`data_pb.csv` and `data_mb.csv`) are placed in the `data/` directory.
2. Run the main preparation script:
```bash
python prep_main.py
```

## Future Work
- Implement additional num prep scripts under the `prep/num/` directory.
- Combine Powerball and Mega Millions data into `num_combined.csv` and perform combined analyses.
- Documents will be build under docs/
- Machine Learning/ AI analysis on chaotic data will be preformed 

## Previous
- Iterations of previous designs and revious can be found under archive
- Archive is a bit of a mess, as this is maining for expiriment and research purposes, rather than results

This README provides an overview of the project, directory structure, file descriptions, setup instructions, and future work. It should help users understand the project and how to use it.