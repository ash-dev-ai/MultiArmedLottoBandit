import pandas as pd
from sklearn.model_selection import train_test_split

# Import the necessary functions and data from FeatureSelection.py
from FeatureSelection import mm_df_split, pb_df_split, univariate_feature_selection

def generate_splits(data_df, target, test_size=0.27, random_state=42):
    # Separate the features and target variable
    X = data_df.drop(columns=[target])
    y = data_df[target]

    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Combine the features and target variable into train and test DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    return train_df, test_df

# Generate train and test splits for MM and PB datasets
mm_train, mm_test = generate_splits(mm_df_split, target='r')
pb_train, pb_test = generate_splits(pb_df_split, target='r')

# Print the train and test splits for MM and PB datasets
print("Train Split for MM Dataset:")
print(mm_train)
print("\nTest Split for MM Dataset:")
print(mm_test)

print("\nTrain Split for PB Dataset:")
print(pb_train)
print("\nTest Split for PB Dataset:")
print(pb_test)
