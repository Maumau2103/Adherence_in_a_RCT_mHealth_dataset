from helper import *
from task2_groups import *
from task1_phases import *
from task3_prediction import *
from task4_statistics import *
import pandas as pd
import warnings

# FutureWarning-Warnungen ignorieren
warnings.filterwarnings("ignore", category=FutureWarning)

s_file_name = 'dataset_sorted.csv'
df_map = find_path(s_file_name)
df_sorted = data_preparation(df_map)

#df_sorted = data_preparation(pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/dataset_sorted.csv', parse_dates=['collected_at']))

result_phases = [17, 46, 81, 100]
allusers_phases = get_allusers_adherence(df_sorted, result_phases)
#newuser_all_phases = [0.9, 0.85, 0.83, 0.8]
#show_user_timeline(df_sorted, 2107, result_phases, start_day=1, end_day=30, step=10)
#allusers_phases = get_allusers_adherence(df_sorted, result_phases)

#non_adherent = get_user_adh_level(df_sorted, 1, start_day=20, end_day=130)
#show_user_adherence_percentage(newuser_all_phases)

#clusters_timeline = cluster_timelines(df_sorted)
#clusters_adherence = cluster_adherence_percentages(allusers_phases)
#clusters_notes = cluster_note_timelines(df_sorted)

#print(clusters_timeline)
#print(clusters_adherence)
#print(clusters_notes)

patient_id_2222 = 2222
preprocessed_data = preprocess_data(df_sorted)
allusers_cluster_label, centroids = k_pod(df_sorted, preprocessed_data, k=3)
print(allusers_cluster_label)
print(centroids)
print(len(allusers_cluster_label))

# Berechnung der Summe für jedes 'phases'-Array und Sortieren des DataFrames nach der berechneten Summe
#allusers_phases_sorted = allusers_phases.assign(phases_sum=allusers_phases['phases'].apply(sum)).sort_values(by='phases_sum')

#print(allusers_phases.iloc[0])

#print(allusers_phases[allusers_phases['user_id'] == 40414])
#print(allusers_phases)

#patient_id_2222 = 2222
#patient_data = preprocess_data(df_sorted[df_sorted['user_id'] == patient_id_2222])
#cluster_assignments, centroids = k_pod(patient_data, k=3)

#user_clusters = combine_cluster_assignments(df_sorted[df_sorted['user_id'] == patient_id_2222], allusers_phases, num_clusters=3)
