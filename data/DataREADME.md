# DataREADME.md

## Overview

This document provides a summary of the total number of combinations for `num1` to `num5` (values 1-70) and `numA` (values 1-26), as well as the time required to insert all possible combinations into a database in batches of 300,000 records each. The calculation is based on the observed time intervals between batch insertions.


## Total Combinations

The total number of combinations is calculated as follows:

- `num1`, `num2`, `num3`, `num4`, `num5` can each take any value from 1 to 70.
- `numA` can take any value from 1 to 26.

The formula to calculate the total number of combinations is:
\[ \text{Total Combinations} = 70^5 \times 26 \]

Performing the calculation:
\[ 70^5 = 70 \times 70 \times 70 \times 70 \times 70 = 16,807,000 \]
\[ 16,807,000 \times 26 = 436,982,000 \]

Thus, there are **4,369,820,000** total combinations.


## Batch Insertion Process

The log entries provided give the timestamps for batch insertions, which allows us to calculate the average time interval between batches. Below are the first few entries as an example:

2024-05-14 21:44:57,130 - INFO - Inserted a batch of records
2024-05-14 21:44:57,747 - INFO - Inserted a batch of records
2024-05-14 21:44:58,361 - INFO - Inserted a batch of records

By computing the differences between consecutive timestamps, we determine the time intervals between batch insertions. The average interval is then used to estimate the total time required.


## Average Time Interval Calculation

We convert the timestamps to `datetime` objects and calculate the time intervals in seconds:

```python
from datetime import datetime

# Log timestamps (example entries)
timestamps = [
    "2024-05-14 21:44:57,130", "2024-05-14 21:44:57,747", "2024-05-14 21:44:58,361",
    "2024-05-14 21:44:58,990", ...
]

# Convert to datetime objects
times = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f") for ts in timestamps]

# Calculate time intervals in seconds
intervals = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]

# Calculate average interval
average_interval = sum(intervals) / len(intervals)

From the provided log entries, the average interval between batch insertions is approximately 0.629 seconds.


## Total Time Calculation

Given the total number of combinations and the batch size of 300,000 records, the total number of batches required is:
\[ \text{Total Batches} = \frac{4,369,820,000}{300,000} = 14,566.067 \]

The total time required is:
\[ \text{Total Time (seconds)} = \text{Total Batches} \times \text{Average Interval} \]
\[ \text{Total Time (seconds)} = 14,566.067 \times 0.629 = 9,162.625 \text{ seconds} \]

We convert this into days, hours, minutes, and seconds:
\[ \text{Total Time} = 1 \text{ day}, 1 \text{ hour}, 6 \text{ minutes}, 52 \text{ seconds} \]

## Conclusion

To insert all 4,369,820,000 combinations in batches of 300,000 records each, it will take approximately **1 day, 1 hour, 6 minutes, and 52 seconds** based on the observed average time interval between batch insertions.

## Author

- Ash Mal, AWS Cloud Full-Stack Engineer

