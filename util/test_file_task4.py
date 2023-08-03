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

result_phases = [20, 41, 63, 84]
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

#show_users_clusters(clusters_timeline)
#show_users_clusters(clusters_adherence)
#show_users_clusters(clusters_notes)