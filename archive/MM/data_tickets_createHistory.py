import os
import random
import pandas as pd
from datetime import datetime, timedelta

def generate_ticket():
    section_w = random.sample(range(1, 70), 5)
    section_r = random.randint(1, 26)
    return section_w, section_r

def generate_tickets(num_tickets, start_date, end_date):
    data = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current_date <= end_date:
        ticket_date = current_date.strftime("%Y-%m-%d")
        for _ in range(num_tickets):
            section_w, section_r = generate_ticket()
            data.append({'d': ticket_date, 'w': str(section_w), 'r': section_r})  # Update column names
        current_date += timedelta(days=1)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    num_tickets_per_day = 10000
    start_date = "2014-01-07"
    datasets = []  # List to store datasets
    
    # Create a directory named "tickets" to save the CSV files
    if not os.path.exists("tickets"):
        os.makedirs("tickets")

    for run in range(1, 7):  # Generate datasets six times
        current_date = datetime.now()
        end_date = (current_date + timedelta(days=(run-1))).strftime("%Y-%m-%d")
        dataset = generate_tickets(num_tickets_per_day, start_date, end_date)
        datasets.append(dataset)  # Add each dataset to the list

        # Save the dataset to a CSV file inside the "tickets" folder with the appropriate name
        filename = f'tickets/data_tickets_{run}.csv'
        dataset.to_csv(filename, index=False)

    print("Datasets saved successfully.")

