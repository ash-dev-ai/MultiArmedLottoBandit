import sys
sys.path.append('data/')

import sqlite3
import pandas as pd

# Import data from data_processWin.py
from data.data_processWin import full_pb

# Connect to the database
db_path = 'data/tickets/tickets.db'
con = sqlite3.connect(db_path)
cur = con.cursor()

# Add full_pb to the database
full_pb.to_sql('full_pb', con, if_exists='replace', index=False)

# SQL query to create the combined DataFrame
query = """
SELECT t.*, f.Type AS win_type, f.Desc AS winning_Desc, f.Probability AS winning_Probability, f.Prize AS winning_Prize
FROM tickets AS t
LEFT JOIN full_pb AS f ON t.day_index = f.d AND t.section_r = f.r
"""

# Execute the SQL query and create the new DataFrame
combined_df = pd.read_sql(query, con)

# Save the combined DataFrame to a new table in the database
combined_df.to_sql('tickets_combined', con, if_exists='replace', index=False)

# Count the number of tickets for each win type in the combined DataFrame
win_type_counts = combined_df['win_type'].value_counts().sort_index()

# Optionally, convert to a DataFrame for better formatting
win_type_counts_df = win_type_counts.reset_index()
win_type_counts_df.columns = ['Win Type', 'Count']

print(win_type_counts_df)

# Close the database connection
con.close()