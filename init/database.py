# database.py
import sqlite3
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = self.create_connection()
        self.enable_journaling()

    def create_connection(self):
        """Create a database connection to a SQLite database."""
        try:
            conn = sqlite3.connect(self.db_name)
            logger.info(f"Connected to the database {self.db_name}")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database {self.db_name}: {e}")
            return None

    def enable_journaling(self):
        """Enable database journaling for better data integrity."""
        if self.conn:
            try:
                cursor = self.conn.cursor()
                cursor.execute('PRAGMA journal_mode=WAL')
                logger.info("Database journaling enabled (WAL mode)")
            except sqlite3.Error as e:
                logger.error(f"Error enabling journaling: {e}")

    def create_table(self):
        """Create a table if it does not exist."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS numbers (
                    id INTEGER PRIMARY KEY,
                    num1 INTEGER NOT NULL,
                    num2 INTEGER NOT NULL,
                    num3 INTEGER NOT NULL,
                    num4 INTEGER NOT NULL,
                    num5 INTEGER NOT NULL,
                    numA INTEGER NOT NULL
                )
            ''')
            self.conn.commit()
            logger.info("Table created successfully")
        except sqlite3.Error as e:
            logger.error(f"Error creating table: {e}")

    def insert_data(self, record_generator, batch_size=500000):
        """Insert records into the table in batches."""
        if self.has_records():
            logger.info("Records already exist in the database. Skipping insertion.")
            return

        try:
            cursor = self.conn.cursor()
            batch = []
            for record in record_generator:
                batch.append(record)
                if len(batch) == batch_size:
                    cursor.executemany('''
                        INSERT INTO numbers (num1, num2, num3, num4, num5, numA)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', batch)
                    self.conn.commit()
                    batch.clear()
                    logger.info("Inserted a batch of records")
            if batch:
                cursor.executemany('''
                    INSERT INTO numbers (num1, num2, num3, num4, num5, numA)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', batch)
                self.conn.commit()
                logger.info("Inserted final batch of records")
        except sqlite3.Error as e:
            logger.error(f"Error inserting data: {e}")

    def has_records(self):
        """Check if the table already has records."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM numbers')
            count = cursor.fetchone()[0]
            return count > 0
        except sqlite3.Error as e:
            logger.error(f"Error checking for records: {e}")
            return False

    def fetch_all_combinations(self, batch_size=100000):
        """Fetch all possible number combinations in batches."""
        query = "SELECT num1, num2, num3, num4, num5, numA FROM numbers"
        for chunk in pd.read_sql_query(query, self.conn, chunksize=batch_size):
            yield chunk

    def close_connection(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
