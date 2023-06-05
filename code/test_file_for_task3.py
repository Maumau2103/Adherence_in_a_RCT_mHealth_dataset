# run this file to test your algorithms and functionalities

from helper import *
from task3_prediction import *
from task1_phases import *
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

df_sorted = pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/dataset_sorted.csv', parse_dates=['collected_at'])
df_newuser = pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/new_user.csv', parse_dates=['collected_at'])

# Finde die k-ähnlichsten Nutzer aus dem Datensatz
# similar_users = find_similar_users(df_sorted, df_newuser, 5)

# Ergänze den Datensatz um das Attribut day
# df_prediction = add_day_attribute(df_sorted)

timeline = get_user_timeline(df_sorted, 2107)

print(timeline)


