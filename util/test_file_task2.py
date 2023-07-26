from task2_groups import *
from helper import *
import pandas as pd

df_map = find_path(s_file_name)
df_sorted = group_and_sort(df_map)
test_cluster_timelines = cluster_timelines(df_sorted, num_clusters=3)
