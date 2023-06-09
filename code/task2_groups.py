import numpy as np
from sklearn.cluster import KMeans
from helper import *
from task1_phases import get_user_timeline
from task5_adherence_level import get_user_adh_level


def cluster_timelines(df_sorted, num_clusters=3, start_day=None, end_day=None):
    all_timelines = []

    # Iterate over each unique user ID
    for user_id in df_sorted[s_table_key].unique():
        # Get the user timeline using the existing function
        timeline = get_user_timeline(df_sorted, user_id, start_day, end_day, s_table_sort_by)
        all_timelines.append(timeline)

    # Convert the timelines to a NumPy array
    timelines_data = np.array(all_timelines)

    # Initialize and train K-Means model
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(timelines_data)

    # Cluster labels for the timelines
    timeline_cluster_labels = kmeans.labels_

    return timeline_cluster_labels


def cluster_adherence_levels(df_sorted, num_clusters=3, start_day=None, end_day=None):
    all_adh_levels = []

    # Iterate over each unique user ID
    for user_id in df_sorted[s_table_key].unique():
        # Get the adherence level for each user using the existing function
        adh_level = get_user_adh_level(df_sorted, user_id, start_day, end_day)
        all_adh_levels.append(adh_level)

    # Convert the adherence levels to a NumPy array
    adh_levels_data = np.array(all_adh_levels).reshape(-1, 1)

    # Initialize and train K-Means model
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(adh_levels_data)

    # Cluster labels for the adherence levels
    adherence_cluster_labels = kmeans.labels_

    return adherence_cluster_labels
