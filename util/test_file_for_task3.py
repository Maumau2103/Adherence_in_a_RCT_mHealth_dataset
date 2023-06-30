import datetime

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
day_y = 10
knn = 10
k_fold = 10
df_user_2107 = df[df['user_id'] == new_user_id].copy()
df_user_2107.to_csv("C:/Users/mauri/PycharmProjects/Softwareprojekt/data/user_2107.csv", index=False)

# neuen Nutzer aus dem Datensatz herausfiltern
#df_prediction_filtered = df_prediction[df_prediction['user_id'] != new_user_id]

# Finde die k-ähnlichsten Nutzer aus dem Datensatz und speichere sie in einem neuen DataFrame
#df_similarusers = find_similar_users(df_prediction_filtered, df_newuser, knn)

# Berechnen der Adherence-Wahrscheinlichkeit für den neuen Nutzer mit SVM
#predictions_svm = svm_classification(df_prediction_filtered, df_newuser, day_y, k_fold)

# Berechnen der Adherence-Wahrscheinlichkeit für den neuen Nutzer mit RandomForest
#predictions_rf = RandomForest_classification(df_prediction_filtered, df_newuser, day_y, k_fold)

# tatsächlicher Wert für unseren neuen User
#add_day_y_adherent(df_newuser, 20)
#print("tatsächliche Adherence für diesen Nutzer an dem Tag: " + str(df_newuser.iloc[0]['day_y_adherent']))

# alle Werte müssen numerisch sein, damit der SVM-Klassifikator funktioniert
# Ein SVM-Algorithmus mit allen Daten von similar_users, wobei die user_id rausgenommen wird und vllt. locale und client auch
# RandomForest funktioniert besser
# similarusers nicht nur anhand adherence level heraussuchen, sondern auch anhand der Länge (ältester Tag - jüngster Tag)