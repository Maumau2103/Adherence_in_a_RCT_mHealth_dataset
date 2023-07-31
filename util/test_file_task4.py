from helper import *
from task4_statistics import *
from task2_groups import *
from task1_phases import *
from task3_prediction import *
import pandas as pd
import warnings

# FutureWarning-Warnungen ignorieren
warnings.filterwarnings("ignore", category=FutureWarning)

df_sorted = data_preparation(pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/dataset_sorted.csv', parse_dates=['collected_at']))

result_phases = [20, 41, 63, 84]
newuser_all_phases = [0.9, 0.85, 0.83, 0.8]
#show_user_timeline(df_sorted, 2107, result_phases, start_day=1, end_day=100, step=10)
allusers_phases = get_allusers_adherence(df_sorted, result_phases)

#non_adherent = get_user_adh_level(df_sorted, 1, start_day=20, end_day=130)
show_user_adherence_percentage(newuser_all_phases)