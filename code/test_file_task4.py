from task4_statistics import *
from task2_groups import *

df_sorted = pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/dataset_sorted.csv', parse_dates=['collected_at'])

#show_user_timeline(df_sorted, 2107)
cluster = cluster_timelines(df_sorted)
print(cluster)