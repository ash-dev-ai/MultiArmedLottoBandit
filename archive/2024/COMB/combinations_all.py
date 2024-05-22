# combinations_all.py
import itertools
import sqlite3
import os
import logging

class CombinationProcessor:
    @staticmethod
    def create_combinations_table(conn, table_name='all_combinations'):
        """
        Creates a database table to store combinations if it doesn't already exist.
        """
        try:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    w1 INTEGER,
                    w2 INTEGER,
                    w3 INTEGER,
                    w4 INTEGER,
                    w5 INTEGER,
                    B INTEGER,
                    valid_for_mb INTEGER DEFAULT 0,
                    valid_for_pb INTEGER DEFAULT 0,
                    PRIMARY KEY (w1, w2, w3, w4, w5, B)
                )
            """)
            conn.commit()
            logging.info(f"Table {table_name} created successfully.")
        except sqlite3.Error as e:
            logging.error(f"Failed to create table {table_name}: {e}")
            raise

    @staticmethod
    def insert_combinations_to_db(conn, combinations_generator, table_name='all_combinations', batch_size=100000):
        """
        Inserts combinations into the database in batches, skipping duplicates.
        """
        cursor = conn.cursor()
        batch = []
        try:
            for comb in combinations_generator:
                cursor.execute(f"""
                    INSERT OR IGNORE INTO {table_name} (w1, w2, w3, w4, w5, B) 
                    VALUES (?, ?, ?, ?, ?, ?)
                """, comb)
                batch.append(comb)
                if len(batch) >= batch_size:
                    conn.commit()
                    batch = []
            # Insert any remaining combinations in the batch
            if batch:
                conn.commit()
        except sqlite3.OperationalError as e:
            logging.error(f"Failed to commit batch: {e}")
            raise
        finally:
            cursor.close()

    @staticmethod
    def generate_all_combinations(w_range, b_range):
        """
        Generates all possible combinations of w1-w5 within w_range and B within b_range.
        """
        for comb in itertools.product(range(*w_range), repeat=5):
            for b in range(*b_range):
                yield comb + (b,)

    @staticmethod
    def main():
        # Define the database file path under the ./data directory
        db_file_path = "./data/all_combinations.db"
        
        # Ensure the data directory exists
        os.makedirs("data", exist_ok=True)

        try:
            # Connect to the database
            conn = sqlite3.connect(db_file_path)
            logging.info("Database connection established.")

            # Create the combinations table
            CombinationProcessor.create_combinations_table(conn)

            # Generate and insert combinations for Mega Millions (MB) into the database
            all_combs_generator_mb = CombinationProcessor.generate_all_combinations((1, 70), (1, 25))
            CombinationProcessor.insert_combinations_to_db(conn, all_combs_generator_mb)

            # Generate and insert combinations for Powerball (PB) into the database
            all_combs_generator_pb = CombinationProcessor.generate_all_combinations((1, 69), (1, 26))
            CombinationProcessor.insert_combinations_to_db(conn, all_combs_generator_pb)

            logging.info("All combinations inserted successfully.")
        except Exception as e:
            logging.error(f"An error occurred in the main execution: {e}")
            raise
        finally:
            # Close the database connection
            conn.close()

if __name__ == "__main__":
    logging.basicConfig(filename='combinations_all.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    CombinationProcessor.main()
