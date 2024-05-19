# prep_main.py
import logging
from prep.prep_pb import PrepPB, pb_data
from prep.prep_mb import PrepMB, mb_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Prepare Powerball data
    pb_preparer = PrepPB()
    pb_preparer.prepare_data()
    if pb_data is not None:
        logging.info(f"Powerball data prepared: {pb_data.head()}")
    else:
        logging.error("Powerball data is None")

    # Prepare Mega Millions data
    mb_preparer = PrepMB()
    mb_preparer.prepare_data()
    if mb_data is not None:
        logging.info(f"Mega Millions data prepared: {mb_data.head()}")
    else:
        logging.error("Mega Millions data is None")

if __name__ == "__main__":
    main()