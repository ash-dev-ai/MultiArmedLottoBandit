# init_main.py
import logging
from itertools import product
from init.config import DATABASE_NAME
from init.database import Database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Initializer:
    def __init__(self, db_name):
        self.db = Database(db_name)

    def run(self):
        if self.db.conn is not None:
            self.db.create_table()
            records_generator = self.generate_all_records()
            self.db.insert_data(records_generator)
            self.db.close_connection()
            logging.info("Database initialized and records generated successfully")
        else:
            logging.error("Failed to create database connection")

    def generate_all_records(self):
        """Generate all possible records with values in the specified ranges."""
        range_70 = range(1, 71)
        range_26 = range(1, 27)
        for record in product(range_70, repeat=5):
            for last_number in range_26:
                yield record + (last_number,)

if __name__ == "__main__":
    initializer = Initializer(DATABASE_NAME)
    initializer.run()

