import random
import sqlite3
from datetime import datetime, timedelta

# Function to generate a single lottery ticket
def generate_ticket():
    w = random.sample(range(1, 70), 5)  # Generate 5 white ball numbers
    r = random.randint(1, 27)  # Generate 1 red ball number
    return w, r

if __name__ == "__main__":
    num_tickets_per_day = 141421  # square root of 2
    start_date = "2010-02-03"  # Starting date for ticket generation
    current_date = datetime.now()  # Get the current date
    
    # Create or connect to an SQLite database
    conn = sqlite3.connect("dbs/tickets.db")
    cursor = conn.cursor()

    # Create a table to store the ticket data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY,
            date TEXT,
            d INTEGER,
            w TEXT,
            w1 INTEGER,
            w2 INTEGER,
            w3 INTEGER,
            w4 INTEGER,
            w5 INTEGER,
            r INTEGER
        )
    ''')

    # Initialize the start date and day index (d)
    current = datetime.strptime(start_date, "%Y-%m-%d")
    d = 1
    
    # Prepare a list to hold batch insert data
    batch_data = []

    # Generate tickets from the start date to the current date
    while current <= current_date:
        ticket_date = current.strftime("%Y-%m-%d")
        
        for _ in range(num_tickets_per_day):
            w, r = generate_ticket()  # Generate a new ticket

            # Add the ticket data to the batch list
            batch_data.append((ticket_date, d, str(w), w[0], w[1], w[2], w[3], w[4], r))

        # Move to the next day and update the day index
        current += timedelta(days=1)
        d += 1

        # Commit the changes to the database after accumulating a certain number of records
        if len(batch_data) >= 100000:  # Change this number based on your requirements
            cursor.executemany('INSERT INTO tickets (date, d, w, w1, w2, w3, w4, w5, r) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', batch_data)
            conn.commit()
            batch_data = []

    # Commit any remaining changes
    if batch_data:
        cursor.executemany('INSERT INTO tickets (date, d, w, w1, w2, w3, w4, w5, r) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', batch_data)
        conn.commit()
    
    # Close the database connection
    conn.close()
    
    print("Data saved successfully to the database.")