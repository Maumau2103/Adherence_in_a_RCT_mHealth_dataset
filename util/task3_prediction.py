import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from task5_adherence_level import *
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt


def task3_prediction(df, new_user, day_y, result_phases, nearest_neighbors=10, cv=10, model=0):
    df_prediction = data_preparation(df)
    df_newuser = data_preparation(new_user)

    # Finde die k-ähnlichsten Nutzer aus dem Datensatz und speichere sie in einem neuen DataFrame
    df_similarusers = find_similar_users(df_prediction, df_newuser, nearest_neighbors)

    if model == 0:
        # Berechnen der Adherence-Wahrscheinlichkeit für den neuen Nutzer mit RandomForest
        predictions_rf = rf_classification(df_similarusers, df_newuser, day_y, cv)
    else:
        # Berechnen der Adherence-Wahrscheinlichkeit für den neuen Nutzer mit SVM
        predictions_svm = svm_classification(df_similarusers, df_newuser, day_y, cv)


def data_preparation(df):
    # Löschen aller Spalten, die nur NULL-Werte enthalten
    df = df.dropna(axis='columns', how = 'all')

    # Anlegen einer drop_list mit allen Spalten, die nicht benötigt werden
    drop_list = ['created_at', 'updated_at', 'collected_at_loudness', 'collected_at_cumberness', 'collected_at_jawbone',
                 'collected_at_neck', 'collected_at_tin_day', 'collected_at_tin_cumber', 'collected_at_tin_max',
                 'collected_at_movement', 'collected_at_stress', 'collected_at_emotion', 'collected_at_diary_q11']
    df = df.drop(drop_list, axis=1)

    # Umwandeln der object-Werte mithilfe des OneHotEncoders
    #encoder = OneHotEncoder()
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


def get_newusers_adherence(df_newuser, result_phases):
    # user_id und adherence_level vom neuen Nutzer
    newuser_id = df_newuser.iloc[0, 1]
    newuser_length = df_newuser['day'].max()
    print('new users id: ' + str(newuser_id))
    print('new users length: ' + str(newuser_length) + ' days')
    print()

    # adherence percentage für alle Phasen herausfinden
    last_change_point = 1
    phases = []
    for i in range(len(result_phases)):
        if (newuser_length > last_change_point):
            print(last_change_point)
            print(result_phases[i])
            adh_percentage = get_user_adh_percentage(df_newuser, newuser_id, last_change_point, result_phases[i])
            phases.append(adh_percentage)
            last_change_point = result_phases[i]

    print(result_phases)
    print(phases)


def find_similar_users(df_prediction, df_newuser, result_phases, k):
    # user_id und adherence_level vom neuen Nutzer
    newuser_id = df_newuser.iloc[0, 1]
    newuser_length = df_newuser['day'].max()
    print('new users id: ' + str(newuser_id))
    print('new users length: ' + str(newuser_length) + ' days')
    print()

    # adherence percentage für alle Phasen herausfinden
    phase_counter = 0
    last_change_point = 1
    phases = []
    for i in range(len(result_phases)):
        if (newuser_length > last_change_point):
            adh_percentage = get_user_adh_percentage(df_newuser, newuser_id, last_change_point, i)
            phases.append(adh_percentage)
            last_change_point = i

    # Initialisierung eines leeren DataFrames
    df_adh_levels = pd.DataFrame(columns=['user_id', 'adherence_level'])

    # Iteration über die eindeutigen user_ids
    for user_id in df_prediction['user_id'].unique():
        # Erstellen einer Zeile mit user_id und adherence_level
        row = {'user_id': user_id, 'adherence_level': get_user_adh_percentage(df_prediction, user_id)}

        # Hinzufügen der Zeile zum Ergebnis-DataFrame
        df_adh_levels = df_adh_levels.append(row, ignore_index=True)

    # Extrahieren der Merkmale für das k-NN-Modell
    features = df_adh_levels[['adherence_level']]

    # Initialisierung des k-NN-Modells
    model = NearestNeighbors(n_neighbors=k)
    model.fit(features)

    # Vorhersage der k ähnlichsten Nutzer für den neuen Nutzer
    new_user_features = [[newuser_adh_level]]
    distances, indices = model.kneighbors(new_user_features)

    # Extrahieren der ähnlichsten Nutzer aus dem ursprünglichen Datensatz
    similar_users = df_adh_levels.iloc[indices[0]]

    print(f"Die {k} ähnlichsten Nutzer sind:")
    print(similar_users)
    print()

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


def svm_classification(df_similarusers, df_newuser, day_y, k_fold):
    # Hinzufügen des day_y_adherent Attributs
    df_similarusers = add_day_y_adherent(df_similarusers, day_y)
    newuser_adh_level = get_user_adh_percentage(df_newuser, df_newuser.iloc[0,1])

    # Entfernen aller unnötigen Spalten (alle kategorischen Attribute)
    columns_to_remove = ['collected_at', 'user_id', 'id', 'client', 'day', 'locale']
    df_similarusers_filtered = df_similarusers.drop(columns=columns_to_remove)
    df_newuser_filtered = df_newuser.drop(columns=columns_to_remove)

    # Extrahiere Attribute und Zielvariablen
    X = df_similarusers_filtered.drop('day_y_adherent', axis=1)
    X_scaled = scale_data_euclidian(X)
    y = df_similarusers_filtered['day_y_adherent']

    # Anzahl der Beispiele für jede Klasse zählen
    class_counts = y.value_counts()
    unique_values = class_counts.unique()

    if len(unique_values) == 1:
        if unique_values[0] == 0:
            adherence_probability = (0 + newuser_adh_level) / 2
            print(f"Adherencewahrscheinlichkeit an Tag {day_y}: {adherence_probability:.2f}")
        elif unique_values[0] == 1:
            adherence_probability = (1 + newuser_adh_level) / 2
            print(f"Adherencewahrscheinlichkeit an Tag {day_y}: {adherence_probability:.2f}")
        return 0

    # Verhältnis der Klassen berechnen
    class_ratio = class_counts[0] / class_counts[1]

    # SVM-Modell initialisieren und Accuracy mit cross validation testen
    svm_model = SVC(class_weight={0: 1.0, 1: class_ratio})
    scores = cross_val_score(svm_model, X_scaled, y, cv=k_fold)
    result = sum(scores) / len(scores)
    print(f"Durchschnittliche Test Accuracy SVM-Modell: {result:.3f}")

    # Trainiere den SVM-Klassifikator
    svm_model.fit(X_scaled, y)

    # Vorhersagen für den neuen Datensatz machen
    predictions = svm_model.predict(df_newuser_filtered)
    adherence_probability = ((sum(predictions) / len(predictions)) + newuser_adh_level) / 2

    print(f"Adherencewahrscheinlichkeit an Tag {day_y}: {adherence_probability:.2f}")

    return predictions


def rf_classification(df_similarusers, df_newuser, day_y, k_fold):
    # Hinzufügen des day_y_adherent Attributs (Label)
    df_similarusers = add_day_y_adherent(df_similarusers, day_y)
    newuser_adh_level = get_user_adh_percentage(df_newuser, df_newuser.iloc[0,1])

    # Entfernen aller unnötigen Spalten (alle kategorischen Attribute)
    columns_to_remove = ['collected_at', 'user_id', 'id', 'client', 'day', 'locale']
    df_similarusers_filtered = df_similarusers.drop(columns=columns_to_remove)
    df_newuser_filtered = df_newuser.drop(columns=columns_to_remove)

    # Datensatz aufteilen in Features und Label
    X = df_similarusers_filtered.drop('day_y_adherent', axis=1)
    y = df_similarusers_filtered['day_y_adherent']

    # Anzahl der Beispiele für jede Klasse zählen
    class_counts = y.value_counts()
    unique_values = class_counts.unique()

    if len(unique_values) == 1:
        if unique_values[0] == 0:
            adherence_probability = (0 + newuser_adh_level) / 2
            print(f"Adherencewahrscheinlichkeit an Tag {day_y}: {adherence_probability:.2f}")
        elif unique_values[0] == 1:
            adherence_probability = (1 + newuser_adh_level) / 2
            print(f"Adherencewahrscheinlichkeit an Tag {day_y}: {adherence_probability:.2f}")
        return 0

    # Verhältnis der Klassen berechnen
    class_ratio = class_counts[0] / class_counts[1]

    # RandomForest-Modell initialisieren und Accuracy mit cross validation testen
    rf_model = RandomForestClassifier(random_state=42, class_weight={0: 1.0, 1: class_ratio})
    scores = cross_val_score(rf_model, X, y, cv=k_fold)
    result = sum(scores) / len(scores)
    print(f"Durchschnittliche Test Accuracy RandomForest-Modell: {result:.3f}")

    # Trainiere den RandomForest-Klassifikator
    rf_model.fit(X, y)

    # Vorhersagen für den neuen Datensatz machen
    predictions = rf_model.predict(df_newuser_filtered)
    adherence_probability = ((sum(predictions) / len(predictions)) + newuser_adh_level) / 2

    print(f"Adherencewahrscheinlichkeit an Tag {day_y}: {adherence_probability:.2f}")

    return predictions


def scale_data_euclidian(X):
    # Skaliere die Daten mithilfe der euklidischen Norm (StandardScaler)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X))

    return X_scaled


# mithilfe von Task 1 herausfinden in welcher Phase sich der Nutzer befindet
# mithilfe von Task 5 herausfinden in welcher Phase er wie adherent war
# Datensatz nach Nutzern filtern, die gleiches Adherence Verhalten aufweisen (k --> Anzahl bestimmen)
# Einträge der Nutzer labeln nach day_y_adherent
# RF-Model auf den gelabelten Daten trainieren
# Accuracy für das Modell berechnen (unbalancierte Daten beachten)
# Adherence Wahrscheinlichkeit angeben
