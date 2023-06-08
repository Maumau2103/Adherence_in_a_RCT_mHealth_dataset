# run this file to test your algorithms and functionalities

from helper import *
from task3_prediction import *
from task1_phases import *
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/gefiltert.csv', parse_dates=['collected_at'])

# Bringe den Datensatz in die Form, dass er von den ML Algorithmen benutzt werden kann (sprich nur numerische Werte etc.)
df_prediction = data_preparation(df)

# Anlegen eines Test-Users mit Werten aus dem bereits vorhandenen Datensatz
#df_newuser = df_prediction[df_prediction['user_id'] == 2107].copy()

# Finde die k-ähnlichsten Nutzer aus dem Datensatz und speichere sie in einem neuen DataFrame
#df_similarusers = find_similar_users(df_prediction, df_newuser, 5)

# Füge die Attribute
#df_similarusers = add_attributes(df_similarusers, 20)
#print(df_similarusers)

#classifiers = svm_classification(df_similarusers, df_newuser)

#predictions = prediction(classifiers, df_newuser)

# alle Werte müssen numerisch sein, damit der SVM-Klassifikator funktioniert
