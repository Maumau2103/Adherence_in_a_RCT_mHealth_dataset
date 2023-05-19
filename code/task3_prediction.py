import pandas as pd
import numpy as np
from sklearn import svm

# Laden der Daten
data = pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/dataset_sorted.csv', parse_dates=['collected_at'])
other_user_data = pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/new_user.csv', parse_dates=['collected_at'])

# Gruppieren der Daten nach Nutzer-ID
grouped_data = data.groupby('user_id')

# Berechnen der Adherence für jeden Nutzer
adherence = {}
for user_id, group in grouped_data:
    # Sortieren der Daten nach collected_at
    group = group.sort_values(by='collected_at')

    # Berechnen der Zeitdifferenzen zwischen den Datensätzen
    diffs = np.diff(group['collected_at'])

    if len(diffs) > 0:
        # Zählen der nicht-aktiven Tage (d.h. Tage, an denen keine Daten erfasst wurden)
        inactive_days = np.sum(diffs > np.timedelta64(1, 'D'))

        # Berechnen der Adherence als Anteil der aktiven Tage
        adherence[user_id] = 1 - (inactive_days / len(diffs))

# Berechnen der Adherence für den anderen Nutzer
other_user_adherence = 1 - (
            np.sum(np.diff(other_user_data['collected_at']) > np.timedelta64(1, 'D')) / len(other_user_data))

# Sortieren der Nutzer nach Ähnlichkeit (d.h. nach Abstand zur Adherence des anderen Nutzers)
similar_users = sorted(adherence.keys(), key=lambda x: abs(adherence[x] - other_user_adherence))[:5]

# Ausgabe der ähnlichen Nutzer
print("Die fünf ähnlichsten Nutzer sind:")
for user_id in similar_users:
    print(f"- Nutzer {user_id} mit Adherence {adherence[user_id]}")

def adding_day_y_adherent(df_similarusers):
    # return df_similarusers, aber mit dem Attribut day_y_adherent
    return df_similarusers

def find_similar_users(df_sorted, df_newuser, day_y, k):
    # return df_similarusers (DataFrame von allen Daten zu den k-ähnlichsten Nutzern)
    return 0

def svm_classification(df_similarusers):
    # Extrahiere Attribute und Zielvariablen
    attributes = df_similarusers.iloc[:, :-1]
    labels = df_similarusers.iloc[:, -1]

    # Initialisiere den SVM-Klassifikator
    classifier = svm.SVC()

    # Trainiere den Klassifikator
    classifier.fit(attributes, labels)

    # Gib den trainierten Klassifikator zurück
    return classifier
