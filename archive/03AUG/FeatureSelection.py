import time
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# Function for Feature Selection using Univariate Feature Selection
def univariate_feature_selection(data_df, target):
    # Convert all column names to strings
    data_df.columns = data_df.columns.astype(str)

    # Separate the target variable from the features
    X = data_df.drop(columns=[target])
    y = data_df[target]

    # Univariate Feature Selection using ANOVA F-value test (for numerical features)
    # Select the top three features based on the ANOVA F-value test
    k_best_f_classif = SelectKBest(score_func=f_classif, k=5)
    X_selected = k_best_f_classif.fit_transform(X, y)

    # Get the selected feature indices
    selected_indices = k_best_f_classif.get_support(indices=True)

    # Return the selected features DataFrame
    return data_df.iloc[:, selected_indices]

# Import data from gravityDrop
from gravityDrop import mm, pb

# Drop 'm' column from MM Dataset and split 'w' into individual columns, then set 'd' as index
mm_df_split = pd.concat([mm.drop(['w', 'm'], axis=1), mm['w'].apply(pd.Series)], axis=1).set_index('d')

# Drop 'm' column from PB Dataset and split 'w' into individual columns, then set 'd' as index
pb_df_split = pd.concat([pb.drop(['w', 'm'], axis=1), pb['w'].apply(pd.Series)], axis=1).set_index('d')

# Perform Univariate Feature Selection on MM Dataset
selected_mm_df = univariate_feature_selection(mm_df_split, target='r')

# Perform Univariate Feature Selection on PB Dataset
selected_pb_df = univariate_feature_selection(pb_df_split, target='r')

# Print the selected MM and PB DataFrames after feature selection
print("Updated MM Dataset after Univariate Feature Selection:")
print(selected_mm_df)

print("\nUpdated PB Dataset after Univariate Feature Selection:")
print(selected_pb_df)

# Record the start time
start_time = time.process_time()

# Calculate and display the elapsed time
elapsed_time = time.process_time() - start_time
print("\nElapsed Time: {:.4f} seconds".format(elapsed_time))


