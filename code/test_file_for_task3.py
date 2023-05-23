# run this file to test your algorithms and functionalities

from helper import *
from task3_prediction import *
import pandas as pd

df_sorted = pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/dataset_sorted.csv', parse_dates=['collected_at'])
df_newuser = pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/new_user.csv', parse_dates=['collected_at'])

# Beispiel-Nutzerdaten als DataFrame
users_data = pd.DataFrame({
    'Attribut1': [1, 4, 7, 10],
    'Attribut2': [2, 5, 8, 11],
    'Attribut3': [3, 6, 9, 12],
    'Zielvariable': ['nein', 'ja', 'nein', 'nein']
})

# Rufe die Funktion auf, um den Klassifikator zu trainieren
classifier = svm_classification(users_data)

# Beispielvorhersage für neue Daten als DataFrame
new_user = pd.DataFrame({
    'Attribut1': [13],
    'Attribut2': [14],
    'Attribut3': [15]
})

prediction(classifier, new_user)

df_sorted_day = add_day_attribute(df_sorted)
print(df_sorted_day)
day_y = 20
df_sorted_day_dayyadherent = add_day_y_adherent(df_sorted_day, day_y)
print(df_sorted_day_dayyadherent)

df_sorted_day_dayyadherent.to_csv("C:/Users/mauri/PycharmProjects/Softwareprojekt/data/dataset_for_prediction.csv", index=False)
