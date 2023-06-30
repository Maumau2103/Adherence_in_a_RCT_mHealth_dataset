import numpy as np
from sklearn.cluster import KMeans

from task1_phases import get_user_timeline, get_all_user_timelines
from task5_adherence_level import get_user_adh_percentage


def cluster_timelines(df_sorted, num_clusters=3, column_name='collected_at', start_day=None, end_day=None):
    timelines = get_all_user_timelines(df_sorted, start_day, end_day)

    # Convert the timelines to a NumPy array
    timelines_data = np.array(timelines)

    # Initialize and train K-Means model
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(timelines_data)

    # Cluster labels for the timelines
    timeline_cluster_labels = kmeans.labels_

    return timeline_cluster_labels


def cluster_adherence_levels(df_sorted, num_clusters=3, start_day=None, end_day=None):
    all_adh_levels = []

    # Iterate over each unique user ID
    for user_id in df_sorted['user_id'].unique():
        # Get the adherence level for each user using the existing function
        adh_level = get_user_adh_percentage(df_sorted, user_id, start_day, end_day)
        all_adh_levels.append(adh_level)

    # Convert the adherence levels to a NumPy array
    adh_levels_data = np.array(all_adh_levels).reshape(-1, 1)

    # Initialize and train K-Means model
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(adh_levels_data)

    # Cluster labels for the adherence levels
    adherence_cluster_labels = kmeans.labels_

    return adherence_cluster_labels


def cluster_note_timelines(df_sorted, num_clusters=4, column_name='collected_at', start_day=None, end_day=None):
    timelines = get_all_user_timelines(df_sorted, start_day, end_day, column_name)
    notes = get_all_user_timelines(df_sorted, start_day, end_day, 'value_diary_q11')

    # Convert the timelines and notes to NumPy arrays
    timelines_data = np.array(timelines)
    notes_data = np.array(notes)

    # Create pairs of binary arrays (timelines and notes)
    paired_data = np.column_stack((timelines_data, notes_data))

    # Initialize and train K-Means model
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(paired_data)

    # Cluster labels for the paired data
    timeline_notes_cluster_labels = kmeans.labels_

    return timeline_notes_cluster_labels
