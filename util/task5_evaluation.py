from task5_adherence_level import *
from task5_adherence_level_helper import *


def adh_level_evaluation(df_sorted, full_adh_threshold=80, non_adh_threshold=40, start_day=s_start_day, end_day=s_end_day):
    all_adh_level_means = []
    adh_level_1 = get_user_adh_level(df_sorted,1,full_adh_threshold,non_adh_threshold,start_day,end_day)
    adh_level_2 = get_user_adh_level(df_sorted, 2, full_adh_threshold, non_adh_threshold, start_day, end_day)
    adh_level_3 = get_user_adh_level(df_sorted, 3, full_adh_threshold, non_adh_threshold, start_day, end_day)

    adh_level_1_mean = 0
    for id in adh_level_1:
        adh_level_1_mean += get_user_adh_percentage(df_sorted,id,start_day,end_day)
    adh_level_1_mean = adh_level_1_mean / len(adh_level_1)

    adh_level_2_mean = 0
    for id in adh_level_2:
        adh_level_2_mean += get_user_adh_percentage(df_sorted, id, start_day, end_day)
    adh_level_2_mean = adh_level_2_mean / len(adh_level_2)

    adh_level_3_mean = 0
    for id in adh_level_3:
        adh_level_3_mean += get_user_adh_percentage(df_sorted, id, start_day, end_day)
    adh_level_3_mean = adh_level_3_mean / len(adh_level_3)
    #adh_level_means_sum = adh_level_1_mean + adh_level_2_mean + adh_level_3_mean
    all_adh_level_means.append(adh_level_1_mean )
    all_adh_level_means.append(adh_level_2_mean )
    all_adh_level_means.append(adh_level_3_mean )

    return all_adh_level_means


def adh_level_cluster_evaluation(df_sorted, start_day=s_start_day, end_day=s_end_day):
    all_adh_level_means = []
    adh_level_1 = get_user_adh_level_cluster(df_sorted,1,start_day,end_day)
    adh_level_2 = get_user_adh_level_cluster(df_sorted,2,start_day,end_day)
    adh_level_3 = get_user_adh_level_cluster(df_sorted,3,start_day,end_day)

    adh_level_1_mean = 0
    for id in adh_level_1:
        adh_level_1_mean += get_user_adh_percentage(df_sorted,id,start_day,end_day)
    adh_level_1_mean = adh_level_1_mean / len(adh_level_1)

    adh_level_2_mean = 0
    for id in adh_level_2:
        adh_level_2_mean += get_user_adh_percentage(df_sorted, id, start_day, end_day)
    adh_level_2_mean = adh_level_2_mean / len(adh_level_2)

    adh_level_3_mean = 0
    for id in adh_level_3:
        adh_level_3_mean += get_user_adh_percentage(df_sorted, id, start_day, end_day)
    adh_level_3_mean = adh_level_3_mean / len(adh_level_3)
    #adh_level_means_sum = adh_level_1_mean + adh_level_2_mean + adh_level_3_mean
    all_adh_level_means.append(adh_level_1_mean )
    all_adh_level_means.append(adh_level_2_mean )
    all_adh_level_means.append(adh_level_3_mean )

    return all_adh_level_means


def adh_level_group_average(df_sorted, full_adh_threshold=80, non_adh_threshold=40, start_day=s_start_day, end_day=s_end_day):

    adh_level_1 = get_user_adh_level(df_sorted, 1, full_adh_threshold, non_adh_threshold, start_day, end_day)
    adh_level_2 = get_user_adh_level(df_sorted, 2, full_adh_threshold, non_adh_threshold, start_day, end_day)
    adh_level_3 = get_user_adh_level(df_sorted, 3, full_adh_threshold, non_adh_threshold, start_day, end_day)

    all_dist_1to2 = 0
    for i1 in adh_level_1:
        for i2 in adh_level_2:
            dist = get_user_adh_percentage(df_sorted,i2,start_day,end_day) - get_user_adh_percentage(df_sorted,i1,start_day,end_day)
            all_dist_1to2 += dist

    all_dist_2to3 = 0
    for i2 in adh_level_2:
        for i3 in adh_level_3:
            dist = get_user_adh_percentage(df_sorted,i3,start_day,end_day) - get_user_adh_percentage(df_sorted,i2,start_day,end_day)
            all_dist_2to3 += dist

    all_dist_1to3 = 0
    for i1 in adh_level_1:
        for i3 in adh_level_3:
            dist = get_user_adh_percentage(df_sorted, i3, start_day, end_day) - get_user_adh_percentage(df_sorted, i1,start_day,end_day)
            all_dist_1to3 += dist

    group_average_1to2 = all_dist_1to2 / (len(adh_level_1) * len(adh_level_2))
    group_average_2to3 = all_dist_2to3 / (len(adh_level_2) * len(adh_level_3))
    group_average_1to3 = all_dist_1to3 / (len(adh_level_1) * len(adh_level_3))

    all_group_averages = [group_average_1to2, group_average_2to3, group_average_1to3]

    return all_group_averages


def adh_level_group_cluster_average(df_sorted, start_day=s_start_day, end_day=s_end_day):

    adh_level_1 = get_user_adh_level_cluster(df_sorted, 1, start_day, end_day)
    adh_level_2 = get_user_adh_level_cluster(df_sorted, 1, start_day, end_day)
    adh_level_3 = get_user_adh_level_cluster(df_sorted, 1, start_day, end_day)

    all_dist_1to2 = 0
    for i1 in adh_level_1:
        for i2 in adh_level_2:
            dist = get_user_adh_percentage(df_sorted,i2,start_day,end_day) - get_user_adh_percentage(df_sorted,i1,start_day,end_day)
            all_dist_1to2 += dist

    all_dist_2to3 = 0
    for i2 in adh_level_2:
        for i3 in adh_level_3:
            dist = get_user_adh_percentage(df_sorted,i3,start_day,end_day) - get_user_adh_percentage(df_sorted,i2,start_day,end_day)
            all_dist_2to3 += dist

    all_dist_1to3 = 0
    for i1 in adh_level_1:
        for i3 in adh_level_3:
            dist = get_user_adh_percentage(df_sorted, i3, start_day, end_day) - get_user_adh_percentage(df_sorted, i1,start_day,end_day)
            all_dist_1to3 += dist

    group_average_1to2 = all_dist_1to2 / (len(adh_level_1) * len(adh_level_2))
    group_average_2to3 = all_dist_2to3 / (len(adh_level_2) * len(adh_level_3))
    group_average_1to3 = all_dist_1to3 / (len(adh_level_1) * len(adh_level_3))

    all_group_averages = [group_average_1to2, group_average_2to3, group_average_1to3]

    return all_group_averages


def comp(df_sorted, full_adh_threshold=80, non_adh_threshold=40, start_day=s_start_day, end_day=s_end_day) :
    non_adherent = get_user_adh_level(df_sorted, 1, full_adh_threshold, non_adh_threshold, start_day, end_day)
    low_adherence = get_user_adh_level_cluster(df_sorted, 1, start_day, end_day)
    non_adh_perc = []
    for i in non_adherent:
        x = get_user_adh_percentage(df_sorted,i,start_day,end_day)
        non_adh_perc.append(x)
    low_adh_perc = []
    for i in low_adherence:
        x = get_user_adh_percentage(df_sorted,i,start_day,end_day)
        low_adh_perc.append(x)
    fully_adherent = get_user_adh_level(df_sorted, 3, full_adh_threshold, non_adh_threshold, start_day, end_day)
    high_adherence = get_user_adh_level_cluster(df_sorted, 3, start_day, end_day)
    full_adh_perc = []
    for i in fully_adherent:
        x = get_user_adh_percentage(df_sorted, i, start_day, end_day)
        full_adh_perc.append(x)
    high_adh_perc = []
    for i in high_adherence:
        x = get_user_adh_percentage(df_sorted,i,start_day,end_day)
        high_adh_perc.append(x)
    non_adh_perc.sort()
    low_adh_perc.sort()
    full_adh_perc.sort()
    high_adh_perc.sort()
    print(max(non_adh_perc))
    print(max(low_adh_perc))
    print(min(full_adh_perc))
    print(min(high_adh_perc))
