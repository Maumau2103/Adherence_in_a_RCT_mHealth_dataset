import pandas as pd
import numpy as np
from sklearn import svm
from helper import *


def find_similar_users(df_sorted, df_newuser, day_y, k):
    # return df_similarusers (DataFrame von allen Daten zu den k-ähnlichsten Nutzern)
    return 0


def add_day_y_adherent(df_similarusers, y):
    # Initialisieren des day_y_adherent-Attributs als False
    df_similarusers['day_y_adherent'] = False

    # Iteration über die Daten
    for user_id, group in df_similarusers.groupby('user_id'):
        # Überprüfen, ob der Tag y für den Nutzer vorhanden ist
        if y in group['day'].values:
            # Setzen des day_y_adherent-Attributs auf True
            df_similarusers.loc[data['user_id'] == user_id, 'day_y_adherent'] = True

    return df_similarusers


def svm_classification(df_similarusers, df_newuser):
    df_newuser_day = add_day_attribute(df_newuser)
    days = get_all_days(df_newuser_day)
    classifiers = []

    for day in days:
        df_similarusers_filtered = filter_by_day(df_similarusers, day)
        classifiers.append(svm_classification_helper(df_similarusers_filtered))

    return classifiers


def svm_classification_helper(df_similarusers):
    # Extrahiere Attribute und Zielvariablen
    attributes = df_similarusers.iloc[:, :-1]
    labels = df_similarusers.iloc[:, -1]

    # Initialisiere den SVM-Klassifikator
    classifier = svm.SVC()

    # Trainiere den Klassifikator
    classifier.fit(attributes, labels)

    # Gib den trainierten Klassifikator zurück
    return classifier


def prediction(svm_classifier, df_newuser):
    # Anwenden des trainierten Klassifikators auf den neuen Nutzer
    prediction = svm_classifier.predict(df_newuser)

    # Ausgeben des Ergebnisses
    print("Vorhersage:", prediction)

    # Gib das Ergebnis zurück
    return prediction


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
