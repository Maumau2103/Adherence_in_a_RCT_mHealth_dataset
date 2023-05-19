# run this file to test your algorithms and functionalities

from helper import *
from task3_prediction import *

dataset = 'C:/Users/mauri/PycharmProjects/Softwareprojekt/data/dataset_sorted.csv'
new_user = 'C:/Users/mauri/PycharmProjects/Softwareprojekt/data/new_user.csv'
day_y = 20

# Einlesen der Datensätze als pandas DataFrame
df_sorted = csv_to_dataframe(dataset)
new_user_df = csv_to_dataframe(new_user)

prediction(df_sorted, new_user_df)
