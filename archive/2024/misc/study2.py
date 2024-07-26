from steps import df_steps 
from sklearn.cluster import KMeans
from datetime import datetime

def study_clustering():
    # Log the start time
    start_time = datetime.now()
    print(f"Script started at: {start_time}")

    df_steps = calculate_steps()
    kmeans = KMeans(n_clusters=3)
    df_steps['cluster'] = kmeans.fit_predict(df_steps)
    print(df_steps['cluster'].value_counts())

    # Log the end time
    end_time = datetime.now()
    print(f"Script ended at: {end_time}")
    print(f"Total time taken: {end_time - start_time}")

if __name__ == "__main__":
    study_clustering()