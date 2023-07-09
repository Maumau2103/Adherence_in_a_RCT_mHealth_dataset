from task1_phases import *
from setup import *

def get_user_adh_percentage(df_sorted, user_id, start_day=None, end_day=None, column=s_table_sort_by):

    # get timeline array from Task 1
    timeline = get_user_timeline(df_sorted, user_id, start_day, end_day, column)

    # create adherence level variable
    adh_percentage = float(0)
    # calculate adherence level for user in timeline
    for index in timeline:
        adh_percentage += index

    adh_percentage = adh_percentage / len(timeline) # percentage

    return adh_percentage

def get_user_adh_level(adh_level, full_adh_threshold=80, non_adh_threshold=40, start_day=None, end_day=None) : # adh_level of 1=non-adherent, 2=partial, 3=full
    adherence_group = []

    for user_id in df_sorted['user_id'].unique():
        if adh_level == 1:
            adh_percentage = get_user_adh_percentage(df_sorted, user_id, start_day, end_day)
            if adh_percentage < non_adh_threshold:
                adherence_group.append(adh_percentage)
        elif adh_level == 2:
            adh_percentage = get_user_adh_percentage(df_sorted, user_id, start_day, end_day)
            if adh_percentage >= non_adh_threshold and adh_percentage < full_adh_threshold:
                adherence_group.append(adh_percentage)
        else:
            adh_percentage = get_user_adh_percentage(df_sorted, user_id, start_day, end_day)
            if adh_percentage >= full_adh_threshold:
                adherence_group.append(adh_percentage)

    return adherence_group
