import pandas as pd
from sklearn.cluster import KMeans
import itertools
from data_makeUniform import data_tickets_com_1, data_tickets_com_2, data_tickets_com_3, full_mm

def create_group(group_name, w_count, has_r=False):
    w_columns = [f'w{i}' for i in range(1, 6)]
    combinations = []
    for w_combination in itertools.combinations(w_columns, w_count):
        combination = list(w_combination)
        if has_r:
            combination.append('r')
        combinations.append(combination)
    return combinations

def create_label(group_name, w_count, has_r=False):
    label = f"{w_count}w"
    if has_r:
        label += "+r"
    return label

# Create an empty DataFrame to store the labeled data
labeled_all_tickets = pd.DataFrame()

# Process data_tickets_com_1 in chunks and classify each entry
chunk_size = 100000
for chunk in pd.read_csv(data_tickets_com_1, chunksize=chunk_size):
    chunk['label'] = ""
    for index, row in chunk.iterrows():
        ticket_numbers = set(row[['w1', 'w2', 'w3', 'w4', 'w5', 'r']])
        for group_name, group_combinations in label_groups.items():
            for combination in group_combinations:
                if set(combination) == ticket_numbers:
                    chunk.at[index, 'label'] = group_name
                    break
    labeled_all_tickets = pd.concat([labeled_all_tickets, chunk])

# Process data_tickets_com_2 in chunks and classify each entry
for chunk in pd.read_csv(data_tickets_com_2, chunksize=chunk_size):
    chunk['label'] = ""
    for index, row in chunk.iterrows():
        ticket_numbers = set(row[['w1', 'w2', 'w3', 'w4', 'w5', 'r']])
        for group_name, group_combinations in label_groups.items():
            for combination in group_combinations:
                if set(combination) == ticket_numbers:
                    chunk.at[index, 'label'] = group_name
                    break
    labeled_all_tickets = pd.concat([labeled_all_tickets, chunk])

# Process data_tickets_com_3 in chunks and classify each entry
for chunk in pd.read_csv(data_tickets_com_3, chunksize=chunk_size):
    chunk['label'] = ""
    for index, row in chunk.iterrows():
        ticket_numbers = set(row[['w1', 'w2', 'w3', 'w4', 'w5', 'r']])
        for group_name, group_combinations in label_groups.items():
            for combination in group_combinations:
                if set(combination) == ticket_numbers:
                    chunk.at[index, 'label'] = group_name
                    break
    labeled_all_tickets = pd.concat([labeled_all_tickets, chunk])

# Perform K-means clustering for org_mm
num_clusters = len(label_groups)
kmeans_org = KMeans(n_clusters=num_clusters, random_state=42)
full_mm["cluster"] = kmeans_org.fit_predict(full_mm.drop("d", axis=1))

# Extract clusters for kmeans_all_tickets
kmeans_all_tickets_1 = labeled_all_tickets[labeled_all_tickets["label"] == "1w+r"]
kmeans_all_tickets_2 = labeled_all_tickets[labeled_all_tickets["label"] == "2w+r"]
kmeans_all_tickets_3 = labeled_all_tickets[labeled_all_tickets["label"] == "3w"]

# Process kmeans_all_tickets_1 through kmeans_all_tickets_3 and return results
kmeans_org_1 = full_mm[full_mm["cluster"] == 0]
kmeans_org_2 = full_mm[full_mm["cluster"] == 1]
kmeans_org_3 = full_mm[full_mm["cluster"] == 2]

# Return the results
kmeans_all_tickets_1, kmeans_all_tickets_2, kmeans_all_tickets_3, kmeans_org_1, kmeans_org_2, kmeans_org_3
