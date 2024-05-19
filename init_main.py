# main.py
import logging
from itertools import product
from init.config import DATABASE_NAME
from init.database import Database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_all_records():
    """Generate all possible records with values in the specified ranges."""
    range_70 = range(1, 71)
    range_26 = range(1, 27)
    for record in product(range_70, range_70, range_70, range_70, range_70, range_26):
        yield record

if __name__ == "__main__":
    db = Database(DATABASE_NAME)
    if db.conn is not None:
        db.create_table()
        records_generator = generate_all_records()
        db.insert_data(records_generator)
        db.close_connection()
        logging.info("Database initialized and records generated successfully")
    else:
        logging.error("Failed to create database connection")

