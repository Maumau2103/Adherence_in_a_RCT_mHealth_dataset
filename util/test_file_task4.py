from helper import *
from task4_statistics import *
from task2_groups import *
import pandas as pd

df_sorted = data_preparation(pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/dataset_sorted.csv', parse_dates=['collected_at']))

#show_user_timeline(df_sorted, 2107)
df_stats = show_user_statistics(df_sorted, 2107)
print(df_stats)