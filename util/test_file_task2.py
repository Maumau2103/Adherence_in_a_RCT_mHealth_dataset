from task2_groups import *
from helper import *
import pandas as pd

# df_map = find_path(s_file_name)
# df_sorted = group_and_sort(df_map)
# test_cluster_timelines = cluster_timelines(df_sorted, num_clusters=3)

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from task2_groups import *


# Load the df_sorted DataFrame from the file
s_file_name = 'dataset_sorted.csv'
df_map = find_path(s_file_name)
df_sorted = group_and_sort(df_map)

user_id = 2222
patient_data = df_sorted[df_sorted['user_id'] == user_id]

# Eindeutige Benutzer-IDs abrufen
user_ids = df_sorted['user_id'].unique()

# Jedem Benutzer basierend auf ihren Daten von mehreren Tagen ein einzelnes Cluster zuweisen
user_clusters = {}

for user_id in user_ids:
    # Daten für den aktuellen Benutzer
    user_data = df_sorted[df_sorted['user_id'] == user_id][selected_attributes].to_numpy()

    # Daten des Benutzers vorverarbeiten
    data = preprocess_data(pd.DataFrame(user_data, columns=selected_attributes))

    # Anzahl der gewünschten Cluster
    num_clusters = 3

    # Clustering durchführen
    cluster_assignments, centroids = k_pod(data, num_clusters)

    # Dem Benutzer dieselbe Cluster-Bezeichnung zuweisen
    user_clusters[user_id] = cluster_assignments[0]

# Benutzer-Cluster ausgeben
print("Benutzer-Cluster:")
for user_id, cluster in user_clusters.items():
    print(f"Benutzer-ID: {user_id}, Cluster: {cluster}")


# Add the cluster assignments to the patient_data DataFrame
timeline_cluster_labels = cluster_timelines(patient_data)
adherence_cluster_labels = cluster_adherence_levels(patient_data)
timeline_notes_cluster_labels = cluster_note_timelines(patient_data)

print("Timeline Cluster Labels:", timeline_cluster_labels)
print("Adherence Cluster Labels:", adherence_cluster_labels)
print("Timeline & Notes Cluster Labels:", timeline_notes_cluster_labels)

