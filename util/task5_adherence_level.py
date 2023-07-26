from task1_phases import *
from task2_groups import *
from setup import *

def get_user_adh_percentage(df_sorted, user_id, start_day=None, end_day=None, column=s_table_sort_by):

    # get timeline array from Task 1
    timeline = get_user_timeline(df_sorted, user_id, start_day, end_day, column)

    # create adherence percentage variable
    adh_percentage = float(0)
    # calculate adherence percentage for user in timeline
    for index in timeline:
        adh_percentage += index

    adh_percentage = adh_percentage / len(timeline) # percentage

    return adh_percentage

def get_user_adh_level(df_sorted, adh_level, full_adh_threshold=80, non_adh_threshold=40, start_day=None, end_day=None) :
    # adh_level of 1=non-adherent, 2=partial, 3=full
    adherence_group = []
    # check for the adh_level the user chose and put the users that fit that criteria in the adherence_group array
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

def get_user_adh_level_cluster(df_sorted, adh_level, start_day=None, end_day=None) :
    cluster_levels = cluster_adherence_levels(df_sorted, 3, start_day, end_day)
    adherence_group = []
    for index in cluster_levels:
        if index == adh_level:
            adherence_group.append(index)

    return adherence_group