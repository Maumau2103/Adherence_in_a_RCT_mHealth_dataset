from task2_groups import *
import pandas as pd

df_sorted = pd.read_csv(r'C:\Users\User\PycharmProjects\Adherence_in_a_RCT_mHealth_dataset\data\dataset_sorted.csv')
test_cluster_timelines = cluster_timelines(df_sorted, num_clusters=3)
