import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from util.helper import data_preparation
import ruptures


from task2_groups import *


# Load the df_sorted DataFrame from the file
s_file_name = 'dataset_sorted.csv'
df_map = find_path(s_file_name)
df_sorted = group_and_sort(df_map)

#df_prepared = data_preparation(df_sorted)
user_id = 2222
#patient_data = df_prepared[df_prepared['user_id'] == user_id]
patient_data = df_sorted[df_sorted['user_id'] == user_id]

# Get unique user IDs
user_ids = df_sorted['user_id'].unique()

# Assign a single cluster to each user based on their multi-day data
user_clusters = {}

for user_id in user_ids:
    # Data for the current user
    user_data = df_sorted[df_sorted['user_id'] == user_id][selected_attributes].to_numpy()

    # Preprocess data for the user
    data = preprocess_data(pd.DataFrame(user_data, columns=selected_attributes))

    # Number of desired clusters
    num_clusters = 3

    # Perform clustering
    cluster_assignments, centroids = k_pod(data, num_clusters)

    # Assign the same cluster label and centroids to the user
    user_clusters[user_id] = {
        'cluster_assignment': cluster_assignments[0],
        'centroids': centroids
    }

# Output user clusters
print("User Clusters:")
for user_id, cluster_info in user_clusters.items():
    cluster_assignment = cluster_info['cluster_assignment']
    centroids = cluster_info['centroids']
    print(f"User ID: {user_id}, Cluster: {cluster_assignment}")
    print(f"Centroids for User ID {user_id}:\n{centroids}")

# Add the cluster assignments to the patient_data DataFrame
#timeline_cluster_labels = cluster_timelines(patient_data)
timeline_cluster_labels = cluster_timelines(patient_data, num_clusters=3)
adherence_cluster_labels = cluster_adherence_percentages(patient_data)
timeline_notes_cluster_labels = cluster_note_timelines(patient_data)

print("Timeline Cluster Labels:", timeline_cluster_labels)
print("Adherence Cluster Labels:", adherence_cluster_labels)
print("Timeline & Notes Cluster Labels:", timeline_notes_cluster_labels)





