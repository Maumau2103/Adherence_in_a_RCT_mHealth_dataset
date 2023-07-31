from helper import *
from task3_prediction import *
from task1_phases import *
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import warnings

# FutureWarning-Warnungen ignorieren
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/gefiltert.csv', parse_dates=['collected_at'])

# Bringe den Datensatz in die Form, dass er von den ML Algorithmen benutzt werden kann (sprich nur numerische Werte etc.)
df_prediction = data_preparation(df)

# Anlegen eines Test-Users mit Werten aus dem bereits vorhandenen Datensatz
new_user_id = 2107
day_y = 61
knn = 10
k_fold = 10
df_newuser = df_prediction[df_prediction['user_id'] == new_user_id].copy()
df_newuser.to_csv("C:/Users/mauri/PycharmProjects/Softwareprojekt/data/user_2107.csv", index=False)

# tatsächlicher Wert für unseren neuen User
df_newuser_dayyadherent = df_newuser.copy()
df_newuser_dayyadherent = add_day_y_adherent(df_newuser_dayyadherent, day_y)
print("tatsächliche Adherence für diesen Nutzer an dem Tag: " + str(df_newuser_dayyadherent.iloc[0]['day_y_adherent']))

# Entfernen aller Tage ab day_y für den neuen Nutzer
df_newuser_filtered = df_newuser.copy()
df_newuser_filtered = df_newuser_filtered.drop(df_newuser_filtered[df_newuser_filtered['day'] >= day_y].index)

# neuen Nutzer aus dem Datensatz herausfiltern
df_prediction_filtered = df_prediction[df_prediction['user_id'] != new_user_id]

result_phases = [20, 41, 63, 84]

newusers_phases = get_newusers_adherence(df_newuser_filtered, result_phases)
allusers_phases = get_allusers_adherence(df_prediction_filtered, result_phases)
print()

# ähnliche Nutzer finden basierend auf der Adherence
df_similarusers = find_similar_users(df_prediction_filtered, newusers_phases, allusers_phases, 10)
print()

# Berechnen der Adherence-Wahrscheinlichkeit an Tag y für den neuen Nutzer mit Random Forest
predictions = predict_day_adherence(df_similarusers, df_newuser_filtered, day_y, k_fold, 0)
print()

# Berechnen der Adherence-Wahrscheinlichkeit in den zukünftigen Phasen für den neuen Nutzer
newusers_future_phases = predict_phase_adherence(df_similarusers, allusers_phases, newusers_phases)

### Notizen
# alle Werte müssen numerisch sein, damit der Klassifikator funktioniert
# Klassifikator mit allen Daten von similar_users, wobei die kat. Attribute rausgenommen werden (werden nicht gebraucht,
# weil sie keinen Informationsgewinn bringen)
# RandomForest funktioniert besser