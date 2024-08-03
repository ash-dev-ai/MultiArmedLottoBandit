# num_main.py
import logging
from prep.num.prep_num_pb import PrepNumPB
from prep.num.prep_num_mb import PrepNumMB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    # Prepare number counts for Powerball
    num_pb_preparer = PrepNumPB()
    num_pb_preparer.prepare_counts()
    logging.info("Powerball number counts dataset prepared.")

    # Prepare number counts for Mega Millions
    num_mb_preparer = PrepNumMB()
    num_mb_preparer.prepare_counts()
    logging.info("Mega Millions number counts dataset prepared.")

if __name__ == "__main__":
    main()
