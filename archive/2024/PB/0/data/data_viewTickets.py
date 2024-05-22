import sqlite3

def display_data(query, limit=5, display_columns=False):
    conn = sqlite3.connect("dbs/tickets.db")
    cursor = conn.cursor()
    
    # If the user wants to display column information
    if display_columns:
        cursor.execute(f"PRAGMA table_info(tickets)")
        col_info = cursor.fetchall()
        print("Column Information:")
        print("ID | Name | Type | Not Null | Default Value | Primary Key")
        for col in col_info:
            print(f"{col[0]} | {col[1]} | {col[2]} | {col[3]} | {col[4]} | {col[5]}")
        print("\n---\n")
    
    cursor.execute(query)
    rows = cursor.fetchmany(limit)
    for row in rows:
        print(row)
    
    conn.close()

if __name__ == "__main__":
    # Display the top 5 rows of the database along with column information
    display_data("SELECT * FROM tickets LIMIT 5", display_columns=True)

    print("\n---\n")

    # Display the last 5 rows of the database
    display_data("SELECT * FROM tickets ORDER BY id DESC LIMIT 5")
