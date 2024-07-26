import pandas as pd
from gravityDrop import mm, pb, new_entries_pb, new_entries_mm
from datasetsApi import data_mm, data_pb
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import time

# Record the start time
start_time = time.process_time()

org_mm_svg = data_mm.copy()
org_pb_svg = data_pb.copy()

# Convert datetime to an integer representing days since the earliest entry in the d column
# Assuming the earliest entry is the minimum date in both datasets
earliest_date_mm = org_mm_svg["d"].min()
earliest_date_pb = org_pb_svg["d"].min()

def datetime_to_days_since_earliest(date, earliest_date):
    return (date - earliest_date).days + 1

org_mm_svg["d"] = org_mm_svg["d"].apply(lambda x: datetime_to_days_since_earliest(x, earliest_date_mm))
org_pb_svg["d"] = org_pb_svg["d"].apply(lambda x: datetime_to_days_since_earliest(x, earliest_date_pb))

datasets_mm_pb = mm, pb, new_entries_pb, new_entries_mm, org_mm_svg, org_pb_svg

# Update the 'w' column in all datasets
for dataset in datasets_mm_pb:
    dataset[['w1', 'w2', 'w3', 'w4', 'w5']] = pd.DataFrame(dataset['w'].to_list(), index=dataset.index)
    dataset.drop(columns=['w'], inplace=True)

def bagged_svm(X, y):
    # Create a Bagged Support Vector Machine Classifier
    svm_classifier = BaggingClassifier(base_estimator=SVC(), n_estimators=50, random_state=42)

    # Fit the model to the data
    svm_classifier.fit(X, y)

    # Return the trained Bagged Support Vector Machine Classifier
    return svm_classifier

# Create a list to store the trained Bagged Support Vector Machine models for each target column
bagged_svm_mm = []
bagged_svm_pb = []
bagged_svm_pb_new = []
bagged_svm_mm_new = []
bagged_svm_pb_org = []
bagged_svm_mm_org = []

# Create bagged SVM models for each target column
for col in org_mm_svg.columns:
    svm_mm = bagged_svm(org_mm_svg.drop(columns=[col]), org_mm_svg[col])
    bagged_svm_mm.append(svm_mm)

for col in org_pb_svg.columns:
    svm_pb = bagged_svm(org_pb_svg.drop(columns=[col]), org_pb_svg[col])
    bagged_svm_pb.append(svm_pb)

for col in new_entries_pb.columns:
    svm_pb_new = bagged_svm(new_entries_pb.drop(columns=[col]), new_entries_pb[col])
    bagged_svm_pb_new.append(svm_pb_new)

for col in new_entries_mm.columns:
    svm_mm_new = bagged_svm(new_entries_mm.drop(columns=[col]), new_entries_mm[col])
    bagged_svm_mm_new.append(svm_mm_new)

for col in org_pb_svg.columns:
    svm_pb_org = bagged_svm(org_pb_svg.drop(columns=[col]), org_pb_svg[col])
    bagged_svm_pb_org.append(svm_pb_org)

for col in org_mm_svg.columns:
    svm_mm_org = bagged_svm(org_mm_svg.drop(columns=[col]), org_mm_svg[col])
    bagged_svm_mm_org.append(svm_mm_org)

# Return the results
def run_bagged_svm():
    return bagged_svm_mm, bagged_svm_pb, bagged_svm_pb_new, bagged_svm_mm_new, bagged_svm_pb_org, bagged_svm_mm_org

# Calculate and print the elapsed time
end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"\nElapsed Time: {elapsed_time:.4f} seconds")
