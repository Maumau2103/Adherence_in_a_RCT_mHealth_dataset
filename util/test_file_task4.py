from helper import *
from task4_statistics import *
from task2_groups import *
from task1_phases import *
import pandas as pd

df_sorted = data_preparation(pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/dataset_sorted.csv', parse_dates=['collected_at']))

#show_user_timeline(df_sorted, 2107)
df_stats = show_user_statistics(df_sorted, 2107)
print(df_stats)

timeline = get_user_timeline(df_sorted, 2107)
timeline2 = get_user_timeline_2(df_sorted, 2107, 5, 100)
print(timeline)
print(timeline2)
print(len(timeline))
print(len(timeline2))