from helper import *
from setup import *
from task1_phases import *
from task2_groups import *
from task5_adherence_level import *
from testtest import *
import matplotlib.pylab as plt

df_map = find_path(s_file_name)
df_sorted = group_and_sort(df_map)

# x = get_user_timeline(df_sorted, 40176, start_day=0, end_day=10)

# all_timelines = get_all_user_timelines(df_sorted)
# all_adherence_percentages = get_all_adherence_percentage(all_timelines)
# adh_level = 1

# print(get_user_timeline(df_sorted, 40176))

# print(len(get_user_ids(df_sorted)))
# print(all_adherence_percentages)
# print(cpd_binseg(all_adherence_percentages))
# print(cpd_botupseg(all_adherence_percentages))
# print(cpd_windowseg(all_adherence_percentages))
# print(get_user_adh_level(df_sorted, adh_level, full_adh_threshold=80, non_adh_threshold=40, start_day=s_start_day, end_day=s_end_day))
# print(get_user_adh_level_cluster(df_sorted,adh_level,s_start_day,s_end_day))
print(get_user_adh_level_cluster(df_sorted, 3))

# plt.plot(all_adherence_percentages)
# plt.show()
