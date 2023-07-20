from helper import *
from setup import *
from task1_phases import *
import matplotlib.pylab as plt

df_map = find_path(s_file_name)
df_sorted = group_and_sort(df_map)

x = get_user_timeline(df_sorted, 40176)

all_timelines = get_all_user_timelines(df_sorted)
all_adherence_percentages = get_all_adherence_percentage(all_timelines)

print(len(get_user_ids(df_sorted)))
print(all_adherence_percentages)
print(cpd_binseg(all_adherence_percentages))

plt.plot(all_adherence_percentages)
plt.show()
