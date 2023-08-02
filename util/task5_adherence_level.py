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
    # convert percentage to decimal
    full_adh_threshold = full_adh_threshold / 100
    non_adh_threshold = non_adh_threshold / 100
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
'''
def get_user_adh_level_cluster(df_sorted, adh_level, start_day=s_start_day, end_day=s_end_day):
    # adh_level of 1 = low adherence, 2 = moderate, 3 = high
    cluster_levels = cluster_adherence_levels(df_sorted, 3, start_day, end_day)
    adherence_group = []
    first_three_unique_clusters = []
    saving_indices = []
    # put the first three unique instances of clusters(1,2,3) in an array and save their respective indices(user_ids)
    for i in range(len(cluster_levels)):
        if len(first_three_unique_clusters) == 3:
            break
        if cluster_levels[i] in first_three_unique_clusters:
            continue
        first_three_unique_clusters.append(cluster_levels[i])
        saving_indices.append(i)

    adh_percentages = []
    # calculate the adh_percentage of the saved indices(user_ids)
    for i in saving_indices:
        adh_percentages.append(get_user_adh_percentage(df_sorted, i, start_day, end_day))
    # order the first three unique clusters based on their respective adh_percentage going from lowest to highest
    value_at_0 = first_three_unique_clusters[adh_percentages.index(min(adh_percentages))]
    value_at_2 = first_three_unique_clusters[adh_percentages.index(max(adh_percentages))]
    value_at_1 = 0
    helper_arr = [value_at_0,value_at_2]
    for x in range(1,4):
        if x not in helper_arr:
            value_at_1 = x
            break

    sorted_clusters = [value_at_0, value_at_1, value_at_2]
    # now we now what clusters is the lowest adh_level and what is the highest
    # put all users from a specific cluster in an array based on which adh_level the user wants to have
    for j in range(len(cluster_levels)):
        if cluster_levels[j] == sorted_clusters[adh_level-1]:
            adherence_group.append(j)

    return adherence_group
    
    '''
