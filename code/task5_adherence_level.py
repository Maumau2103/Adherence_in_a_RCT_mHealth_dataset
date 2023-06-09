from task1_phases import *


def get_user_adh_percentage(df_sorted, user_id, start_day=None, end_day=None):

    # get timeline array from Task 1
    timeline = get_user_timeline(df_sorted, user_id, start_day, end_day, 'collected_at')

    # create adherence level variable
    adh_percentage = float(0)
    # calculate adherence level for user in timeline
    for index in timeline:
        adh_percentage += index

    adh_percentage = adh_percentage / len(timeline) # percentage

    return adh_percentage
