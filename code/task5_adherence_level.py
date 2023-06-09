from task1_phases import *


def get_user_adh_level(df_sorted, user_id, start_day=None, end_day=None):

    # get timeline array from Task 1
    timeline = get_user_timeline(df_sorted, user_id, start_day, end_day, s_table_sort_by)

    # create adherence level variable
    adh_level = float(0)
    # calculate adherence level for user in timeline
    for index in timeline:
        adh_level += index

    adh_level = adh_level / len(timeline)

    return adh_level
