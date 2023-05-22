from helper import *
from task1_phases import *
import pandas as pd

def statistics(df_sorted, user_id):
    print("users timeline:")

    timeline = get_user_timeline(df_sorted, user_id)
    days = ['day ' + str(i) for i in range(1, len(timeline)+1)]

    # Eine Liste von Listen erstellen, wobei other_values die einzige Unterliste ist
    data = [timeline]

    # DataFrame erstellen und values als Spaltennamen verwenden
    df_phases = pd.DataFrame(data, columns=days)

    print(df_phases)
    return df_phases