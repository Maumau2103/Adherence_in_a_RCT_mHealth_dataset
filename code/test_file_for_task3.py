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
df_newuser = df_prediction[df_prediction['user_id'] == 40362].copy()
df_newuser.to_csv("C:/Users/mauri/PycharmProjects/Softwareprojekt/data/new_user.csv", index=False)

# Finde die k-ähnlichsten Nutzer aus dem Datensatz und speichere sie in einem neuen DataFrame
df_similarusers = find_similar_users(df_prediction, df_newuser, 10)

# Berechnen der Adherence-Wahrscheinlichkeit für den neuen Nutzer mit SVM
predictions_svm = svm_classification(df_similarusers, df_newuser, 20)
print(predictions_svm)

# Berechnen der Adherence-Wahrscheinlichkeit für den neuen Nutzer mit RandomForest
predictions_rf = RandomForest_classification(df_similarusers, df_newuser, 20)
print(predictions_rf)

# tatsächlicher Wert für unseren neuen User
add_day_y_adherent(df_newuser, 20)
print("tatsächliche Adherence für diesen Nutzer an dem Tag: " + str(df_newuser.iloc[0]['day_y_adherent']))

# alle Werte müssen numerisch sein, damit der SVM-Klassifikator funktioniert
# Ein SVM-Algorithmus mit allen Daten von similar_users, wobei die user_id rausgenommen wird und vllt. locale und client auch