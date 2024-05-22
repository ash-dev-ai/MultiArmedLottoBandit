import os
import sqlite3
from itertools import combinations

# Check if the directory exists, if not create it
db_dir = 'dbs'
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

# Create a new SQLite database
db_path = os.path.join(db_dir, 'permutations.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create a table to store Powerball combinations
cursor.execute("""
CREATE TABLE IF NOT EXISTS permutations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    white_numbers TEXT,
    red_number INTEGER
);
""")
conn.commit()

# Function to generate and insert Powerball combinations in batches
def generate_and_insert_permutations_batched(batch_size=10000):
    # White ball combinations
    white_balls = range(1, 70)  # 1 to 69
    white_combinations = combinations(white_balls, 5)

    # Red ball combinations (Powerball)
    red_balls = range(1, 27)  # 1 to 26
    
    batch = []
    for white_comb in white_combinations:
        for red_ball in red_balls:
            white_str = ','.join(map(str, white_comb))
            batch.append((white_str, red_ball))
            
            if len(batch) >= batch_size:
                cursor.executemany("INSERT INTO permutations (white_numbers, red_number) VALUES (?, ?)", batch)
                conn.commit()
                batch.clear()

    # Insert any remaining records
    if batch:
        cursor.executemany("INSERT INTO permutations (white_numbers, red_number) VALUES (?, ?)", batch)
        conn.commit()

# Generate and insert Powerball combinations in batches
try:
    generate_and_insert_permutations_batched()
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Close the database connection
    conn.close()
