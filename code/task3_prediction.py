import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from helper import *
from task5_adherence_level import *


def data_preparation(df):
    # Löschen aller Spalten, die nur NULL-Werte enthalten
    df = df.dropna(axis='columns', how = 'all')

    # Anlegen einer drop_list mit allen Spalten, die nicht benötigt werden
    drop_list = ['created_at', 'updated_at', 'collected_at_loudness', 'collected_at_cumberness', 'collected_at_jawbone',
                 'collected_at_neck', 'collected_at_tin_day', 'collected_at_tin_cumber', 'collected_at_tin_max',
                 'collected_at_movement', 'collected_at_stress', 'collected_at_emotion', 'collected_at_diary_q11']
    df = df.drop(drop_list, axis=1)

    # Umwandeln der object-Werte
    string_columns = ['locale', 'client']
    df[string_columns] = df[string_columns].astype('str')
    df['value_diary_q11'] = df['value_diary_q11'].apply(lambda x: 1 if isinstance(x, str) else 0)

    return df


def find_similar_users(df_sorted, df_newuser, k):
    # Initialisierung eines leeren DataFrames
    df_adh_levels = pd.DataFrame(columns=['user_id', 'adherence_level'])

    # user_id und adherence_level vom neuen Nutzer
    newuser_id = df_newuser.iloc[0,1]
    newuser_adh_level = get_user_adh_percentage(df_newuser, newuser_id)

    # Iteration über die eindeutigen user_ids
    for user_id in df_sorted['user_id'].unique():
        # Erstellen einer Zeile mit user_id und adherence_level
        row = {'user_id': user_id, 'adherence_level': get_user_adh_percentage(df_sorted, user_id)}

        # Hinzufügen der Zeile zum Ergebnis-DataFrame
        df_adh_levels = df_adh_levels.append(row, ignore_index=True)

    # Extrahieren der Merkmale für das k-NN-Modell
    features = df_adh_levels['adherence_level'].values.reshape(-1, 1)

    # Initialisierung des k-NN-Modells
    model = NearestNeighbors(n_neighbors=k)
    model.fit(features)

    # Vorhersage der k ähnlichsten Nutzer für den neuen Nutzer
    new_user_adherence = [[newuser_adh_level]]
    distances, indices = model.kneighbors(new_user_adherence)

    # Extrahieren der ähnlichsten Nutzer aus dem ursprünglichen Datensatz
    similar_users = df_adh_levels.iloc[indices[0]]

    print("Die " + str(k) + " ähnlichsten Nutzer sind:")
    print(similar_users)

    # Herausfiltern von allen similar_users aus df_sorted
    df_similarusers = df_sorted[df_sorted['user_id'].isin(similar_users['user_id'])]

    return df_similarusers


def add_attributes(df_similarusers, y):
    add_day_attribute(df_similarusers)
    add_day_y_adherent(df_similarusers, y)

    return df_similarusers


def add_day_y_adherent(df_similarusers, y):
    # Initialisieren des day_y_adherent-Attributs als False
    df_similarusers['day_y_adherent'] = False

    # Iteration über die Daten
    for user_id, group in df_similarusers.groupby('user_id'):
        # Überprüfen, ob der Tag y für den Nutzer vorhanden ist
        timeline = get_user_timeline(df_similarusers, user_id)
        if timeline[y] == 1:
            df_similarusers.loc[df_similarusers['user_id'] == user_id, 'day_y_adherent'] = True
        #if y in group['day'].values:
            # Setzen des day_y_adherent-Attributs auf True
            #df_similarusers.loc[df_similarusers['user_id'] == user_id, 'day_y_adherent'] = True

    return df_similarusers


def svm_classification(df_similarusers, df_newuser):
    df_newuser_day = add_day_attribute(df_newuser)
    days = df_newuser_day['day'].unique().tolist()
    classifiers = []

    for day in days:
        df_similarusers_filtered = df_similarusers[df_similarusers['day'] == day]
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


def prediction(svm_classifiers, df_newuser):
    predictions = []

    df_newuser_day = add_day_attribute(df_newuser)
    days = df_newuser_day['day'].unique().tolist()

    for day in days:
        df_newuser_day_filtered = df_newuser_day[df_newuser_day['day'] == day]
        prediction = svm_classifiers[day].predict(df_newuser_day_filtered)
        predictions.append(prediction)

    # Ausgeben des Ergebnisses
    print("Vorhersage:", predictions)

    # Gib das Ergebnis zurück
    return predictions
