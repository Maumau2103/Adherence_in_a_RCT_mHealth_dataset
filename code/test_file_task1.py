from task1_phases import *
from task3_prediction import *
import ruptures as rpt

# Alle User Timeslines
test_task1 = get_all_user_timelines(grouped_data)

# prozentuale Verteilung aller User Timelines
perc_test_task1 = calculate_percentage(test_task1)

print(perc_test_task1)

signal, bkps = rpt.pw_constant(500, 2, 3, noise_std=4)

print(type(signal))