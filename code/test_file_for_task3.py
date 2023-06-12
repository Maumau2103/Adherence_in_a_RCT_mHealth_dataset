# run this file to test your algorithms and functionalities

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
df_newuser = df_prediction[df_prediction['user_id'] == 2107].copy()

# Finde die k-ähnlichsten Nutzer aus dem Datensatz und speichere sie in einem neuen DataFrame
df_similarusers = find_similar_users(df_prediction, df_newuser, 5)

# Hinzufügen des day_y_adherent Attributs
df_similarusers = add_day_y_adherent(df_similarusers, 20)

# Anlegen von SVM_Klassifikatoren
classifiers = svm_classification(df_similarusers, df_newuser)

# Berechnen der predictions
#predictions = prediction(classifiers, df_newuser)

# alle Werte müssen numerisch sein, damit der SVM-Klassifikator funktioniert
# Ein SVM-Algorithmus mit allen Daten von similar_users, wobei die user_id rausgenommen wird und vllt. locale und client auch