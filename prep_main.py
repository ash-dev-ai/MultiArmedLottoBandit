# prep_main.py
import logging
from prep.prep_pb import PrepPB
from prep.prep_mb import PrepMB
from prep.stats import DataStatistics
from prep.sums import DataSums
from prep.days import DataDays
from prep.export import DataExporter
from prep.steps import DataSteps
from prep.prep_num_pb import PrepNumPB
from prep.prep_num_mb import PrepNumMB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    # Prepare Powerball data
    pb_preparer = PrepPB()
    pb_preparer.prepare_data()
    pb_data = pb_preparer.get_data()
    if pb_data is not None:
        logging.info(f"Powerball data prepared: {pb_data.head()}")

        # Calculate stats
        stats_processor = DataStatistics(pb_data, 'pb_data')
        pb_data = stats_processor.process()

        # Add sums
        sums_processor = DataSums(pb_data, 'pb_data')
        pb_data = sums_processor.add_sums()

        # Add day and date columns
        days_processor = DataDays(pb_data, 'pb_data')
        pb_data = days_processor.process()

        # Add step columns
        steps_processor = DataSteps(pb_data, 'pb_data')
        pb_data = steps_processor.add_difference_columns()

        logging.info(f"Powerball data with stats, sums, days, dates, and steps: {pb_data.head()}")
    else:
        logging.error("Powerball data is None")

    # Prepare Mega Millions data
    mb_preparer = PrepMB()
    mb_preparer.prepare_data()
    mb_data = mb_preparer.get_data()
    if mb_data is not None:
        logging.info(f"Mega Millions data prepared: {mb_data.head()}")

        # Calculate stats
        stats_processor = DataStatistics(mb_data, 'mb_data')
        mb_data = stats_processor.process()

        # Add sums
        sums_processor = DataSums(mb_data, 'mb_data')
        mb_data = sums_processor.add_sums()

        # Add day and date columns
        days_processor = DataDays(mb_data, 'mb_data')
        mb_data = days_processor.process()

        # Add step columns
        steps_processor = DataSteps(mb_data, 'mb_data')
        mb_data = steps_processor.add_difference_columns()

        logging.info(f"Mega Millions data with stats, sums, days, dates, and steps: {mb_data.head()}")
    else:
        logging.error("Mega Millions data is None")
    
    # Export datasets
    if pb_data is not None and mb_data is not None:
        exporter = DataExporter(pb_data, mb_data)
        exporter.export_to_csv()
    
    # Prepare number counts for Powerball
    if pb_data is not None:
        num_pb_preparer = PrepNumPB()
        num_pb_preparer.prepare_counts()
        logging.info("Powerball number counts dataset prepared.")
    
    # Prepare number counts for Mega Millions
    if mb_data is not None:
        num_mb_preparer = PrepNumMB()
        num_mb_preparer.prepare_counts()
        logging.info("Mega Millions number counts dataset prepared.")

if __name__ == "__main__":
    main()

