# run this file to test your algorithms and functionalities

from helper import *
from task3_prediction import *
import pandas as pd

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
prediction = classifier.predict(new_user)

print("Vorhersage:", prediction)
