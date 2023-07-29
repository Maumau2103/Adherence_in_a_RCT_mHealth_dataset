from helper import *
import numpy as np
import ruptures as rpt
from setup import *


def get_user_timeline(df_sorted, key_column, start_day=None, end_day=None, column=s_table_sort_by_alt):
    # Herausfiltern aller Einträge eines spezifischen Nutzers
    user_df = df_sorted[df_sorted[s_table_key] == key_column]

    # Erstellen einer Liste aller Tage basierend auf den optionalen Parametern
    if start_day is None:
        start_day = 1

    if end_day is None:
        end_day = user_df["day"].max()

    # Erstellen eines binären Arrays für die User Timeline
    timeline = []
    for day in range(start_day, end_day+1):
        if day in user_df[column].tolist():
            timeline.append(1)
        else:
            timeline.append(0)

    return timeline


def get_all_user_timelines(df_sorted, start_day=None, end_day=None):
    # Diese Methode soll alle User Timelines in einem mehrdimensionalen Array ausgeben.
    # Eingabewerte: --> siehe Methode "get_user_timeline"

    user_ids = get_user_ids(df_sorted)
    timelines = []

    for i in range(len(user_ids)):
        timeline = get_user_timeline(df_sorted, user_ids[i], start_day=start_day, end_day=end_day, column=s_table_sort_by)
        timelines.append(timeline)

    return timelines


def get_all_adherence_percentage(timelines):
    num_users = len(timelines)
    timeline_length = len(timelines[0])
    all_adherence_percentages = [0] * timeline_length

    for timeline in timelines:
        for i in range(timeline_length):
            if i < len(timeline) and timeline[i] == 1:
                all_adherence_percentages[i] += 1

    for i in range(timeline_length):
        all_adherence_percentages[i] = (all_adherence_percentages[i] / num_users) * 100

    return all_adherence_percentages

def cpd_binseg(all_adherence_percentages):
    percentages = np.array(all_adherence_percentages).flatten()
    algo = rpt.Binseg(model="l1", jump=1)

    if s_cpd_mode:
        result = algo.fit_predict(percentages, n_bkps=s_num_change_points)
    else:
        result = algo.fit_predict(percentages, pen=s_pen_change_points)

    return result

def cpd_botupseg(all_adherence_percentages):
    percentages = np.array(all_adherence_percentages).flatten()
    algo = rpt.BottomUp(model="l1", jump=1)

    if s_cpd_mode:
        result = algo.fit_predict(percentages, n_bkps=s_num_change_points)
    else:
        result = algo.fit_predict(percentages, pen=s_pen_change_points)

    return result

def cpd_windowseg(all_adherence_percentages):
    percentages = np.array(all_adherence_percentages).flatten()
    algo = rpt.Window(width=40, model="l1", jump=1)

    if s_cpd_mode:
        result = algo.fit_predict(percentages, n_bkps=s_num_change_points)
    else:
        result = algo.fit_predict(percentages, pen=s_pen_change_points)

    return result
