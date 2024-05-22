import pandas as pd
from processing_functions import print_dataset_info

org_mm = pd.read_csv('processed_datasets/org_mm_processed.csv')
org_pb = pd.read_csv('processed_datasets/org_pb_processed.csv')
full_mm = pd.read_csv('processed_datasets/full_mm_processed.csv')
full_pb = pd.read_csv('processed_datasets/full_pb_processed.csv')
new_mm = pd.read_csv('processed_datasets/new_mm_processed.csv')
new_pb = pd.read_csv('processed_datasets/new_pb_processed.csv')
ticket_datasets_com_1 = pd.read_csv('tickets/data_tickets_com_1.csv')
ticket_datasets_com_2 = pd.read_csv('tickets/data_tickets_com_2.csv')
ticket_datasets_com_3 = pd.read_csv('tickets/data_tickets_com_3.csv')

print_dataset_info("org_mm", org_mm)
print_dataset_info("org_pb", org_pb)
print_dataset_info("full_mm", full_mm)
print_dataset_info("full_pb", full_pb)
print_dataset_info("new_mm", new_mm)
print_dataset_info("new_pb", new_pb)
print_dataset_info("ticket_datasets_com_1", ticket_datasets_com_1)
print_dataset_info("ticket_datasets_com_2", ticket_datasets_com_2)
print_dataset_info("ticket_datasets_com_3", ticket_datasets_com_3)
