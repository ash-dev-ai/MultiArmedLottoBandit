# prep_main.py
import logging
from prep.prep_pb import PrepPB
from prep.prep_mb import PrepMB
from prep.stats import calculate_stats
from prep.sums import add_sums
from prep.days import add_day_column, add_date_column
from prep.export import export_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    # Prepare Powerball data
    pb_preparer = PrepPB()
    pb_preparer.prepare_data()
    pb_data = pb_preparer.get_data()
    if pb_data is not None:
        logging.info(f"Powerball data prepared: {pb_data.head()}")
        calculate_stats(pb_data, 'pb_data')
        add_sums(pb_data, 'pb_data')
        add_day_column(pb_data, 'pb_data')
        add_date_column(pb_data, 'pb_data')
        logging.info(f"Powerball data with sums, days, and dates: {pb_data.head()}")
    else:
        logging.error("Powerball data is None")

    # Prepare Mega Millions data
    mb_preparer = PrepMB()
    mb_preparer.prepare_data()
    mb_data = mb_preparer.get_data()
    if mb_data is not None:
        logging.info(f"Mega Millions data prepared: {mb_data.head()}")
        calculate_stats(mb_data, 'mb_data')
        add_sums(mb_data, 'mb_data')
        add_day_column(mb_data, 'mb_data')
        add_date_column(mb_data, 'mb_data')
        logging.info(f"Mega Millions data with sums, days, and dates: {mb_data.head()}")
    else:
        logging.error("Mega Millions data is None")
    
    # Export datasets
    if pb_data is not None and mb_data is not None:
        export_data(pb_data, mb_data)

if __name__ == "__main__":
    main()
