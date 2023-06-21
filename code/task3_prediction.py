import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from helper import *
from task5_adherence_level import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def data_preparation(df):
    # Löschen aller Spalten, die nur NULL-Werte enthalten
    df = df.dropna(axis='columns', how = 'all')

    # Anlegen einer drop_list mit allen Spalten, die nicht benötigt werden
    drop_list = ['created_at', 'updated_at', 'collected_at_loudness', 'collected_at_cumberness', 'collected_at_jawbone',
                 'collected_at_neck', 'collected_at_tin_day', 'collected_at_tin_cumber', 'collected_at_tin_max',
                 'collected_at_movement', 'collected_at_stress', 'collected_at_emotion', 'collected_at_diary_q11']
    df = df.drop(drop_list, axis=1)

    # Umwandeln der object-Werte mithilfe des OrdinalEncoders
    #encoder = OrdinalEncoder(dtype='int64')
    #df[['locale', 'client']] = encoder.fit_transform(df[['locale', 'client']])

    # Umwandeln des diary Eintrags
    df['value_diary_q11'] = df['value_diary_q11'].apply(lambda x: 1 if isinstance(x, str) else 0)

    # Umwandeln von collected_at in datetime Objekte
    df['collected_at'] = pd.to_datetime(df['collected_at'])

    # Aufteilung des collected_at Attributs in mehrere Spalten
    df['collected_at_year'] = df['collected_at'].dt.year
    df['collected_at_month'] = df['collected_at'].dt.month
    df['collected_at_time'] = (df['collected_at'].dt.hour * 60 + df['collected_at'].dt.minute)

    # Hinzufügen des day-Attributes
    df = add_day_attribute(df)

    return df


def find_similar_users(df_prediction, df_newuser, k):
    # Initialisierung eines leeren DataFrames
    df_adh_levels = pd.DataFrame(columns=['user_id', 'adherence_level'])

    # user_id und adherence_level vom neuen Nutzer
    newuser_id = df_newuser.iloc[0,1]
    newuser_adh_level = get_user_adh_percentage(df_newuser, newuser_id)

    # Iteration über die eindeutigen user_ids
    for user_id in df_prediction['user_id'].unique():
        # Erstellen einer Zeile mit user_id und adherence_level
        row = {'user_id': user_id, 'adherence_level': get_user_adh_percentage(df_prediction, user_id)}

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
    df_similarusers = df_prediction[df_prediction['user_id'].isin(similar_users['user_id'])]

    return df_similarusers


def add_day_y_adherent(df_similarusers, y):
    # Initialisieren des day_y_adherent-Attributs als False
    df_similarusers['day_y_adherent'] = 0

    # Iteration über die Daten
    for user_id, group in df_similarusers.groupby('user_id'):
        # Überprüfen, ob der Tag y für den Nutzer vorhanden ist
        if y in group['day'].values:
            # Setzen des day_y_adherent-Attributs auf True
            df_similarusers.loc[df_similarusers['user_id'] == user_id, 'day_y_adherent'] = 1

    return df_similarusers


def svm_classification(df_similarusers, df_newuser, day_y):
    # Hinzufügen des day_y_adherent Attributs
    df_similarusers = add_day_y_adherent(df_similarusers, day_y)

    # Entfernen aller unnötigen Spalten (alle kategorischen Attribute)
    columns_to_remove = ['collected_at', 'user_id', 'id', 'client', 'day', 'locale']
    df_similarusers_filtered = df_similarusers.drop(columns=columns_to_remove)
    df_newuser_filtered = df_newuser.drop(columns=columns_to_remove)

    # Extrahiere Attribute und Zielvariablen
    X = df_similarusers_filtered.drop('day_y_adherent', axis=1)
    y = df_similarusers_filtered['day_y_adherent']

    # SVM-Modell initialisieren und Accuracy testen
    svm_model = SVC()
    ml_model_accuracy(X, y, svm_model, 50)

    # Trainiere den SVM-Klassifikator
    svm_model.fit(X, y)

    # Vorhersagen für den neuen Datensatz machen
    predictions = svm_model.predict(df_newuser_filtered)
    print("Adherencewahrscheinlichkeit an Tag " + str(day_y) + ": " + str(sum(predictions) / len(predictions)))

    return predictions


def RandomForest_classification(df_similarusers, df_newuser, day_y):
    # Hinzufügen des day_y_adherent Attributs (Label)
    df_similarusers = add_day_y_adherent(df_similarusers, day_y)

    # Entfernen aller unnötigen Spalten (alle kategorischen Attribute)
    columns_to_remove = ['collected_at', 'user_id', 'id', 'client', 'day', 'locale']
    df_similarusers_filtered = df_similarusers.drop(columns=columns_to_remove)
    df_newuser_filtered = df_newuser.drop(columns=columns_to_remove)

    # Datensatz aufteilen in Features und Label
    X = df_similarusers_filtered.drop('day_y_adherent', axis=1)
    y = df_similarusers_filtered['day_y_adherent']

    # RandomForest-Modell initialisieren und Accuracy testen
    rf_model = RandomForestClassifier(random_state=42)
    ml_model_accuracy(X, y, rf_model, 50)

    # Trainiere den RandomForest-Klassifikator
    rf_model.fit(X, y)

    # Vorhersagen für den neuen Datensatz machen
    predictions = rf_model.predict(df_newuser_filtered)
    print("Adherencewahrscheinlichkeit an Tag " + str(day_y) + ": " + str(sum(predictions)/len(predictions)))

    return predictions
