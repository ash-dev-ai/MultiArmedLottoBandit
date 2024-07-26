import pandas as pd
from gravityDrop import mm, pb, new_entries_pb, new_entries_mm
from datasetsApi import data_mm, data_pb
from sklearn.ensemble import GradientBoostingClassifier
import time
# Record the start time
start_time = time.process_time()

org_mm_bg = data_mm.copy()
org_pb_bg = data_pb.copy()

# Convert datetime to an integer representing days since the earliest entry in the d column
# Assuming the earliest entry is the minimum date in both datasets
earliest_date_mm_bg = org_mm_bg["d"].min()
earliest_date_pb_bg = org_pb_bg["d"].min()

def datetime_to_days_since_earliest(date, earliest_date):
    return (date - earliest_date).days + 1

org_mm_bg["d"] = org_mm_bg["d"].apply(lambda x: datetime_to_days_since_earliest(x, earliest_date_mm_bg))
org_pb_bg["d"] = org_pb_bg["d"].apply(lambda x: datetime_to_days_since_earliest(x, earliest_date_pb_bg))

datasets_bg = mm, pb, new_entries_pb, new_entries_mm, org_mm_bg, org_pb_bg

# Update the 'w' column in all datasets
for dataset in datasets_bg:
    dataset[['w1', 'w2', 'w3', 'w4', 'w5']] = pd.DataFrame(dataset['w'].to_list(), index=dataset.index)
    dataset.drop(columns=['w'], inplace=True)

def boost_gradient(X, y):
    # Create a Gradient Boosting Classifier
    bg_classifier = GradientBoostingClassifier(n_estimators=50, random_state=42)

    # Fit the model to the data
    bg_classifier.fit(X, y)

    # Return the trained Gradient Boosting Classifier
    return bg_classifier

# Create a list to store the trained Gradient Boosting models for each target column
boost_gradient_mm = []
boost_gradient_pb = []
boost_gradient_pb_new = []
boost_gradient_mm_new = []
boost_gradient_pb_org = []
boost_gradient_mm_org = []

# Create boost gradient models for each target column
for col in org_mm_bg.columns:
    bg_mm = boost_gradient(org_mm_bg.drop(columns=[col]), org_mm_bg[col])
    boost_gradient_mm.append(bg_mm)

for col in org_pb_bg.columns:
    bg_pb = boost_gradient(org_pb_bg.drop(columns=[col]), org_pb_bg[col])
    boost_gradient_pb.append(bg_pb)

for col in new_entries_pb.columns:
    bg_pb_new = boost_gradient(new_entries_pb.drop(columns=[col]), new_entries_pb[col])
    boost_gradient_pb_new.append(bg_pb_new)

for col in new_entries_mm.columns:
    bg_mm_new = boost_gradient(new_entries_mm.drop(columns=[col]), new_entries_mm[col])
    boost_gradient_mm_new.append(bg_mm_new)

for col in org_pb_bg.columns:
    bg_pb_org = boost_gradient(org_pb_bg.drop(columns=[col]), org_pb_bg[col])
    boost_gradient_pb_org.append(bg_pb_org)

for col in org_mm_bg.columns:
    bg_mm_org = boost_gradient(org_mm_bg.drop(columns=[col]), org_mm_bg[col])
    boost_gradient_mm_org.append(bg_mm_org)

# Return the results
def run_boost_gradient():
    return boost_gradient_mm, boost_gradient_pb, boost_gradient_pb_new, boost_gradient_mm_new, boost_gradient_pb_org, boost_gradient_mm_org

# Calculate and print the elapsed time
end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time:.4f} seconds")
