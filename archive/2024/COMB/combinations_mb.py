# combinations_mb.py
import sqlite3
import logging

class CombinationProcessor:
    @staticmethod
    def create_historical_table(conn, table_name='historical_mb_data'):
        """
        Creates the historical MB data table if it doesn't exist.
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
                    mb INTEGER
                )
            """)
            conn.commit()
            logging.info(f"Table {table_name} created successfully.")
        except sqlite3.Error as e:
            logging.error(f"Failed to create table {table_name}: {e}")
            raise

    @staticmethod
    def get_historical_data(conn, table_name='historical_mb_data'):
        """
        Fetches historical MB data from the database.
        """
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT w1, w2, w3, w4, w5, mb FROM {table_name}")
            return cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Failed to fetch data from {table_name}: {e}")
            raise

    @staticmethod
    def is_valid_for_mb(comb, historical_data):
        """
        Determines if a combination is valid for MB based on historical data and specific criteria.
        """
        # Avoiding combinations that match historical draws too closely
        for entry in historical_data:
            if len(set(comb[:-1]) & set(entry[:-1])) >= 3:
                return False

        # Checking for consecutive sequences
        sorted_comb = sorted(comb[:-1])  # Ignore the MB number for this check
        consecutive_count = 1
        for i in range(1, len(sorted_comb)):
            if sorted_comb[i] == sorted_comb[i - 1] + 1:
                consecutive_count += 1
                if consecutive_count >= 3:
                    return False
            else:
                consecutive_count = 1

        return True

    @staticmethod
    def batch_update_valid_mb_status(cursor, combinations, historical_data, table_name='all_combinations'):
        """
        Batch updates the 'valid_for_mb' status for combinations in the database.
        """
        params = []
        for comb in combinations:
            is_valid = CombinationProcessor.is_valid_for_mb(comb, historical_data)
            params.append((is_valid, *comb))
        cursor.executemany(f"""
            UPDATE {table_name}
            SET valid_for_mb = ?
            WHERE w1 = ? AND w2 = ? AND w3 = ? AND w4 = ? AND w5 = ? AND mb = ?
        """, params)

    @staticmethod
    def update_valid_mb_status(conn, table_name='all_combinations', historical_table='historical_mb_data'):
        """
        Updates the 'valid_for_mb' status for each combination in the database in batches.
        """
        try:
            historical_data = CombinationProcessor.get_historical_data(conn, historical_table)
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name}")
            batch_size = 250000
            while True:
                combinations = cursor.fetchmany(batch_size)
                if not combinations:
                    break
                CombinationProcessor.batch_update_valid_mb_status(cursor, combinations, historical_data)
                conn.commit()
            logging.info("Database updated successfully.")
        except Exception as e:
            logging.error(f"An error occurred while updating database: {e}")
            raise

    @staticmethod
    def main():
        db_file_path = "./data/all_combinations.db"
        try:
            with sqlite3.connect(db_file_path) as conn:
                logging.info("Database connection established.")
                CombinationProcessor.create_historical_table(conn)
                CombinationProcessor.update_valid_mb_status(conn)
        except Exception as e:
            logging.error(f"An error occurred in the main execution: {e}")

if __name__ == "__main__":
    CombinationProcessor.main()
