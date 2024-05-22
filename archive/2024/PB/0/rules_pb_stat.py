import sqlite3

DB_PATH = 'data/dbs/tickets.db'

def create_indexes(con, cur):
    index_columns = ['win_type', 'Desc', 'Probability', 'Prize']
    for col in index_columns:
        try:
            cur.execute(f"CREATE INDEX idx_{col} ON tickets ({col})")
            con.commit()
        except sqlite3.OperationalError:
            pass  # Index already exists

def summary_statistics():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Create indexes for faster query
    create_indexes(con, cur)

    # Create a dictionary to store summary statistics
    summary_stats = {}

    # List of columns to summarize
    columns_to_summarize = ['win_type', 'Desc', 'Probability', 'Prize']

    for col in columns_to_summarize:
        cur.execute(f"SELECT {col}, COUNT(*) as count FROM tickets GROUP BY {col}")
        summary_stats[col] = cur.fetchall()

    con.close()

    return summary_stats

if __name__ == "__main__":
    summary_stats = summary_statistics()

    for col, stats in summary_stats.items():
        print(f"\n{col} Counts:")
        for value, count in stats:
            print(f"{value}: {count}")