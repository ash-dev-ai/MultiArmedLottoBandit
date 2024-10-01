# body/prep.py
import pandas as pd
import logging
import os

class BodyDataPrep:
    def __init__(self, pb_filepath, mb_filepath, comb_filepath, output_dir):
        self.pb_filepath = pb_filepath
        self.mb_filepath = mb_filepath
        self.comb_filepath = comb_filepath
        self.output_dir = output_dir

    def load_data(self):
        # Load datasets
        self.pb_data = pd.read_csv(self.pb_filepath)
        self.mb_data = pd.read_csv(self.mb_filepath)
        self.comb_data = pd.read_csv(self.comb_filepath)
        logging.info("Datasets loaded successfully")

    def rename_columns(self):
        # Renaming columns for pb (variation_a)
        pb_columns = {'num1': 'x1', 'num2': 'x2', 'num3': 'x3', 'num4': 'x4', 'num5': 'x5', 'numA': 'z'}
        self.pb_data.rename(columns=pb_columns, inplace=True)

        # Renaming columns for Mb (variation_b)
        mb_columns = {'num1': 'x1', 'num2': 'x2', 'num3': 'x3', 'num4': 'x4', 'num5': 'x5', 'numA': 'z'}
        self.mb_data.rename(columns=mb_columns, inplace=True)

        # Renaming columns for Combined (variation_c)
        comb_columns = {'num1': 'x1', 'num2': 'x2', 'num3': 'x3', 'num4': 'x4', 'num5': 'x5', 'numA': 'z'}
        self.comb_data.rename(columns=comb_columns, inplace=True)
        logging.info("Columns renamed successfully")

    def add_day_and_date(self):
        # Adding 'day' and 'date' columns (assuming a 'draw_date' column exists for the date)
        for dataset, name in [(self.pb_data, 'Powerball'), (self.mb_data, 'Mega Millions'), (self.comb_data, 'Combined')]:
            if 'draw_date' in dataset.columns:
                dataset['date'] = pd.to_datetime(dataset['draw_date'])
                
                # Extract day of the week as a number (0 = Monday, ..., 6 = Sunday)
                dataset['day'] = dataset['date'].dt.dayofweek  
                
                # Format the date as YYYYMMDD for a numeric representation
                dataset['date'] = dataset['date'].dt.strftime('%Y%m%d').astype(int)
                
                logging.info(f"Added numeric 'day' and 'date' columns to {name} dataset.")
            else:
                logging.warning(f"'draw_date' column not found in {name} dataset.")

    def select_columns(self, columns_to_copy):
        # Selecting only the specified columns for all datasets
        self.pb_data = self.pb_data[columns_to_copy]
        self.mb_data = self.mb_data[columns_to_copy]
        self.comb_data = self.comb_data[columns_to_copy]
        logging.info(f"Selected columns: {columns_to_copy}")

    def save_prepped_data(self):
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Save the prepped data to the specified output directory
        pb_output = os.path.join(self.output_dir, 'prepped_variation_a.csv')
        mb_output = os.path.join(self.output_dir, 'prepped_variation_b.csv')
        comb_output = os.path.join(self.output_dir, 'prepped_variation_c.csv')

        self.pb_data.to_csv(pb_output, index=False)
        self.mb_data.to_csv(mb_output, index=False)
        self.comb_data.to_csv(comb_output, index=False)
        logging.info(f"Prepped datasets saved to {self.output_dir}")

    def run(self, columns_to_copy):
        # Complete preparation process
        self.load_data()
        self.rename_columns()
        self.add_day_and_date()
        self.select_columns(columns_to_copy)
        self.save_prepped_data()

# Main execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Filepaths for the datasets
    pb_filepath = '../data/data_pb.csv'
    mb_filepath = '../data/data_mb.csv'
    comb_filepath = '../data/data_combined.csv'
    
    # Output directory
    output_dir = './data/'  # Save the new datasets here

    # Columns to copy over (Specify only the ones you want to keep)
    columns_to_copy = ['x1', 'x2', 'x3', 'x4', 'x5', 'z', 'day', 'date']

    # Initialize and run the prep process
    prepper = BodyDataPrep(pb_filepath, mb_filepath, comb_filepath, output_dir)
    prepper.run(columns_to_copy)


