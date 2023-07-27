from helper import *
from task4_statistics import *
from task2_groups import *
from task1_phases import *
import pandas as pd

df_sorted = data_preparation(pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/dataset_sorted.csv', parse_dates=['collected_at']))

result_phases = [20, 41, 63, 84]
show_user_timeline(df_sorted, 2107, result_phases, start_day=1, end_day=100, step=10)