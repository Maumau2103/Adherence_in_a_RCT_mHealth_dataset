from task1_phases import *
from task2_groups import *
from setup import *

def get_user_adh_percentage(df_sorted, user_id, start_day=s_start_day, end_day=s_end_day):

    # get timeline array from Task 1
    timeline = get_user_timeline(df_sorted, user_id, start_day, end_day)

    adh_sum = sum(timeline)
    adh_percentage = adh_sum / len(timeline)

    return adh_percentage

def get_user_adh_level(df_sorted, adh_level, full_adh_threshold=80, non_adh_threshold=40, start_day=s_start_day, end_day=s_end_day) :
    # adh_level of 1=non-adherent, 2=partial, 3=full
    adherence_group = []
    user_ids = get_user_ids(df_sorted)
    # check for the adh_level the user chose and put the users that fit that criteria in the adherence_group array

    if adh_level == 1:
        for i in range(len(user_ids)):
            adh_percentage = get_user_adh_percentage(df_sorted, user_ids[i], start_day, end_day)
            if adh_percentage < non_adh_threshold:
                adherence_group.append(user_ids[i])

    elif adh_level == 2:
        for i in range(len(user_ids)):
            adh_percentage = get_user_adh_percentage(df_sorted, user_ids[i], start_day, end_day)
            if adh_percentage >= non_adh_threshold and adh_percentage < full_adh_threshold:
                adherence_group.append(user_ids[i])

    else:
        for i in range(len(user_ids)):
            adh_percentage = get_user_adh_percentage(df_sorted, user_ids[i], start_day, end_day)
            if adh_percentage >= full_adh_threshold:
                adherence_group.append(user_ids[i])

    return adherence_group

def get_user_adh_level_cluster(df_sorted, adh_level, start_day=None, end_day=None) :
    cluster_levels = cluster_adherence_levels(df_sorted, 3, start_day, end_day)
    adherence_group = []
    for index in cluster_levels:
        if index == adh_level:
            adherence_group.append(index)

    return adherence_group