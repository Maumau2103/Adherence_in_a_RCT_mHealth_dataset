import pandas as pd
import numpy as np
from numpy.linalg import norm
import math
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


def get_newusers_adherence(df_newuser, result_phases):
    # user_id und length vom neuen Nutzer
    newuser_id = df_newuser.iloc[0, 1]
    newuser_length = df_newuser['day'].max()
    print('new users id: ' + str(newuser_id))
    print('new users length: ' + str(newuser_length) + ' days')

    # adherence percentage für alle Phasen herausfinden
    last_change_point = 1
    phases = []
    for change_point in result_phases:
        if (newuser_length > last_change_point):
            adh_percentage = get_user_adh_percentage(df_newuser, newuser_id, last_change_point, change_point)
            phases.append(round(adh_percentage, 3))
            last_change_point = change_point

    if (newuser_length > last_change_point):
        adh_percentage = get_user_adh_percentage(df_newuser, newuser_id, start_day=last_change_point, end_day=None)
        phases.append(round(adh_percentage, 3))

    print('new users phases: ' + str(len(phases)))
    print('new users adherence percentages: ' + str(phases))
    return phases


def get_allusers_adherence(df_sorted, result_phases):
    # adherence percentage für alle Nutzer in allen Phasen herausfinden
    user_phases = pd.DataFrame(columns=["user_id", "phases"])
    for user_id in df_sorted['user_id'].unique().tolist():
        df_user = df_sorted[df_sorted['user_id'] == user_id].copy()
        phases = []
        last_change_point = 1
        user_length = df_user['day'].max()
        for change_point in result_phases:
            if (user_length > last_change_point):
                adh_percentage = get_user_adh_percentage(df_user, user_id, last_change_point, change_point)
                phases.append(round(adh_percentage, 3))
            else:
                phases.append(0)
            last_change_point = change_point

        if (user_length > last_change_point):
            adh_percentage = get_user_adh_percentage(df_user, user_id, start_day=last_change_point, end_day=None)
            phases.append(round(adh_percentage, 3))
        else:
            phases.append(0)

        new_row = {"user_id": user_id, "phases": phases}
        user_phases = user_phases.append(new_row, ignore_index=True)

    return user_phases


def shorten_list(lst, n):
    return lst[:n]


def find_similar_users(df_sorted, new_users_phases, all_users_phases, k):
    # Kürzen von allen phases-Einträgen auf die Länge von new_users_adherence
    df = all_users_phases.copy()
    df['phases'] = df['phases'].apply(lambda x: shorten_list(x, len(new_users_phases)))

    # Berechne die Ähnlichkeiten zwischen der gegebenen Liste und den Listen im DataFrame
    df["similarity"] = df["phases"].apply(lambda x: euclidean_distance(x, new_users_phases))

    # Wähle die k ähnlichsten Listen im DataFrame aus
    similar_users = df.nsmallest(k, "similarity")

    print(f"Die {k} ähnlichsten Nutzer sind:")
    print()
    print(similar_users)

    # Herausfiltern von allen similar_users aus df_sorted
    df_similarusers = df_sorted[df_sorted['user_id'].isin(similar_users['user_id'])]

    return df_similarusers


def euclidean_distance(phases1, phases2):
    return round(math.dist(phases1, phases2), 3)


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


def predict_day_adherence(df_similarusers, df_newuser, day_y, k_fold, model):
    # Hinzufügen des day_y_adherent Attributs
    df_similarusers = add_day_y_adherent(df_similarusers, day_y)
    newuser_adh_percentage = get_user_adh_percentage(df_newuser, df_newuser.iloc[0,1], start_day=None, end_day=None)

    # Entfernen aller kategorischen Attribute
    columns_to_remove = ['collected_at', 'user_id', 'id', 'client', 'day', 'locale']
    df_similarusers_filtered = df_similarusers.drop(columns=columns_to_remove)
    df_newuser_filtered = df_newuser.drop(columns=columns_to_remove)

    # Extrahiere Attribute und Zielvariablen
    X = df_similarusers_filtered.drop('day_y_adherent', axis=1)
    X_scaled = scale_data_euclidian(X)
    y = df_similarusers_filtered['day_y_adherent']

    # Anzahl der Beispiele für jede Klasse zählen
    class_counts = y.value_counts()
    count_ones = 0
    count_zeros = 0

    for element in y:
        if element == 1:
            count_ones += 1
        elif element == 0:
            count_zeros += 1

    # Ausgeben der Adherence Wahrscheinlichkeit, wenn nur eine Klasse des Labels vorhanden ist
    if count_zeros == 0:
        adherence_probability = (1 + newuser_adh_percentage) / 2
        print(f"Adherencewahrscheinlichkeit an Tag {day_y}: {adherence_probability:.2f}")
        return 0
    elif count_ones == 0:
        adherence_probability = (0 + newuser_adh_percentage) / 2
        print(f"Adherencewahrscheinlichkeit an Tag {day_y}: {adherence_probability:.2f}")
        return 0

    # Verhältnis der Klassen berechnen
    class_ratio = class_counts[0] / class_counts[1]

    if model == 0:
        # RandomForest-Modell initialisieren, trainieren und Accuracy mit cross validation testen
        classification_model = RandomForestClassifier(random_state=42, class_weight={0: 1.0, 1: class_ratio})
        scores = cross_val_score(classification_model, X, y, cv=k_fold)
        result = sum(scores) / len(scores)
        print(f"Durchschnittliche Test Accuracy RandomForest-Modell: {result:.3f}")
        classification_model.fit(X, y)
    else:
        # SVM-Modell initialisieren, trainieren und Accuracy mit cross validation testen
        classification_model = SVC(class_weight={0: 1.0, 1: class_ratio})
        scores = cross_val_score(classification_model, X_scaled, y, cv=k_fold)
        result = sum(scores) / len(scores)
        print(f"Durchschnittliche Test Accuracy SVM-Modell: {result:.3f}")
        classification_model.fit(X_scaled, y)

    # Vorhersagen für den neuen Datensatz machen
    predictions = classification_model.predict(df_newuser_filtered)

    # Wahrscheinlichkeit für adherence berechnen (50% vom Modell beeinflusst, 50% von der durchschnittlichen Adherence)
    adherence_probability = ((sum(predictions) / len(predictions)) + newuser_adh_percentage) / 2
    print(f"Adherencewahrscheinlichkeit an Tag {day_y}: {adherence_probability:.3f}")

    return predictions


def shorten_list_2(lst, n):
    if n >= len(lst):
        return lst

    return lst[-n:]


def predict_phase_adherence(df_similarusers, allusers_phases, newuser_phases):
    # Filtere allusers_phases basierend auf den "user_id"-Werten in df_similarusers
    similarusers_phases = allusers_phases[allusers_phases['user_id'].isin(df_similarusers['user_id'])]

    # Anzahl der Phasen, die für den neuen Nutzer in der Zukunft liegen
    new_len = len(allusers_phases.iloc[0]['phases']) - len(newuser_phases)

    # Kürzen von allen phases-Einträgen, sodass nur die Phasen angezeigt werden, die für den neuen Nutzer in der Zukunft liegen
    similarusers_phases['phases'] = similarusers_phases['phases'].apply(lambda x: shorten_list_2(x, new_len))

    newuser_future_phases = []

    for i in range(new_len):
        sum_adh_percentage = 0
        for j in range(len(similarusers_phases)):
            sum_adh_percentage += similarusers_phases.iloc[j]['phases'][i]
        newuser_future_phases.append(round(sum_adh_percentage/len(similarusers_phases), 3))

    print('new users adherence percentages in future phases: ' + str(newuser_future_phases))

    return newuser_future_phases


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
