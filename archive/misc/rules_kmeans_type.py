import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
from fractions import Fraction

# Import data from data_processWin.py
from data.data_processWin import full_pb

# Import winning data from data_winRules.py
from data.data_winRules import winning_data

# Connect to the database
db_path = 'data/tickets/tickets.db'
con = sqlite3.connect(db_path)
cur = con.cursor()

# Define a function to assign types based on matching criteria
def assign_type(row):
    if row['r'] == 1:
        return 9
    elif row['w1'] == 1 and row['w2'] == 1 and row['w3'] == 1 and row['w4'] == 1 and row['w5'] == 1:
        return 8
    elif row['w1'] == 1 or row['w2'] == 1 or row['w3'] == 1 or row['w4'] == 1 or row['w5'] == 1:
        if row['r'] == 1:
            return 7
        else:
            return 6
    elif row['r'] == 1:
        return 5
    else:
        return 0

# Load data from the 'tickets' table in the database
query = "SELECT * FROM tickets"
tickets_df = pd.read_sql_query(query, con)

# Assign types based on matching criteria
tickets_df['Type'] = tickets_df.apply(assign_type, axis=1)

# Prepare the data for clustering
features = ['d', 'w1', 'w2', 'w3', 'w4', 'w5', 'r']
X = tickets_df[features]

# Perform K-means clustering
kmeans = KMeans(n_clusters=10, random_state=0)
tickets_df['Cluster'] = kmeans.fit_predict(X)

# Update the 'Type' and 'Cluster' columns in the database
tickets_df[['Type', 'Cluster']].to_sql('tickets', con, if_exists='replace', index=False)

# Commit changes and close the database connection
con.commit()
con.close()