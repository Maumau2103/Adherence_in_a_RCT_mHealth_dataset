from helper import *
from task1_phases import *

def statistics(df_sorted, user_id):
    timeline = get_user_timeline(df_sorted, user_id)
    print(timeline)