from helper import *
from setup import *
from task1_phases import *
from task2_groups import *
from task5_adherence_level import *
from task5_evaluation import *
import matplotlib.pylab as plt

df_map = find_path(s_file_name)
df_sorted = data_preparation(df_map)

print(get_user_timeline(df_sorted, 2107, 20, 40))
all_timelines = get_all_user_timelines(df_sorted)
all_adherence_percentages = get_all_adherence_percentage(all_timelines)
adh_level = 1

print(get_user_timeline(df_sorted, 40176))

print(len(get_user_ids(df_sorted)))
print(all_adherence_percentages)
print(cpd_binseg(all_adherence_percentages))
print(cpd_botupseg(all_adherence_percentages))
print(cpd_windowseg(all_adherence_percentages))

# plt.plot(all_adherence_percentages)
# plt.show()
