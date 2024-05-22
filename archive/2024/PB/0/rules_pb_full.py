import sys
from fractions import Fraction
import sqlite3
import pandas as pd

sys.path.append('data/')

from data_processWin import full_pb, org_pb
from data_winRules import winning_data

DB_PATH = 'data/dbs/tickets.db'

def add_missing_columns(con, cur):
    for col in ['win_type', 'Desc', 'Probability', 'Prize']:
        try:
            cur.execute(f"ALTER TABLE tickets ADD COLUMN {col}")
            con.commit()
        except sqlite3.OperationalError:
            pass

def get_unique_days(con):
    unique_days_tickets = pd.read_sql("SELECT DISTINCT d FROM tickets", con)['d'].tolist()
    unique_days_full_pb = full_pb['d'].unique().tolist()
    return set(unique_days_tickets).intersection(set(unique_days_full_pb))

def process_tickets_for_day(day):
    print(f"Processing tickets for day: {day}")
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    query = "SELECT * FROM tickets WHERE d = ?"
    tickets_df = pd.read_sql(query, con, params=[day])
    print(f"Loaded {len(tickets_df)} tickets for day {day}")

    win_types = []
    descs = []
    probs = []
    prizes = []

    winning_numbers = full_pb[full_pb['d'] == day].iloc[0]
    print(f"Winning numbers for day {day}: {winning_numbers}")

    for _, ticket in tickets_df.iterrows():
        w_matches = sum(ticket[f'w{i+1}'] == winning_numbers[f'w{i+1}'] for i in range(5))
        r_match = ticket['r'] == winning_numbers['r']

        win_type, desc, prob, prize = determine_win_type(w_matches, r_match)

        win_types.append(win_type)
        descs.append(desc)
        probs.append(prob)
        prizes.append(prize)

    tickets_df['win_type'] = win_types
    tickets_df['Desc'] = descs
    tickets_df['Probability'] = probs
    tickets_df['Prize'] = prizes

    update_query = '''
    UPDATE tickets SET win_type = ?, Desc = ?, Probability = ?, Prize = ? WHERE id = ?
    '''
    cur.executemany(update_query, tickets_df[['win_type', 'Desc', 'Probability', 'Prize', 'id']].values.tolist())
    con.commit()
    con.close()
    print(f"Finished processing tickets for day {day}")

def determine_win_type(w_matches, r_match):
    win_type = 0  # Default to 'no win'
    if w_matches == 5 and r_match:
        win_type = 1
    elif w_matches == 5:
        win_type = 2
    elif w_matches == 4 and r_match:
        win_type = 3
    elif w_matches == 4:
        win_type = 4
    elif w_matches == 3 and r_match:
        win_type = 5
    elif w_matches == 3:
        win_type = 6
    elif w_matches == 2 and r_match:
        win_type = 7
    elif w_matches == 1 and r_match:
        win_type = 8
    elif r_match:
        win_type = 9

    desc, prob, prize = winning_data.get(win_type, ('No Win', 0.0, 0))
    prob = float(prob) if isinstance(prob, Fraction) else prob
    return win_type, desc, prob, prize

# Main script execution starts here
with sqlite3.connect(DB_PATH) as con:
    cur = con.cursor()
    add_missing_columns(con, cur)
    unique_days = get_unique_days(con)

for day in unique_days:
    process_tickets_for_day(day)

with sqlite3.connect(DB_PATH) as con:
    cur = con.cursor()
    org_pb['w'] = org_pb['w'].apply(lambda x: ','.join(map(str, x)))
    org_pb.to_sql('tickets', con, if_exists='append', index=False)
    con.commit()

with sqlite3.connect(DB_PATH) as con:
    cur = con.cursor()
    query = "SELECT * FROM tickets"
    new_tickets_df = pd.read_sql(query, con)

print(f"Number of rows in tickets_df after appending org_pb: {len(new_tickets_df)}")
