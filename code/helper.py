import pandas as pd
import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from setup import *

def group_and_sort(df):
    # Gruppieren des DataFrame nach user_id und Sortieren nach collected_at
    df_sorted = df.groupby(s_table_key).apply(lambda x: x.sort_values([s_table_sort_by], ascending=True)).reset_index(drop=True)
    return df_sorted


def get_user_ids(grouped_data):
    # Die Methode gibt alle User IDs in einem sortierten Array aus.
    user_ids = grouped_data[s_table_key].unique()
    return user_ids


def add_day_attribute(df_sorted):
    # Konvertieren von 'collected_at' in das Datumsformat
    df_sorted[s_table_sort_by] = pd.to_datetime(df_sorted[s_table_sort_by])

    # Initialisieren des Tagesattributs
    df_sorted['day'] = 0

    # Iteration über die Daten
    for user_id, group in df_sorted.groupby(s_table_key):
        # Bestimmung des ältesten Datums pro user_id
        min_date = group[s_table_sort_by].min()

        # Berechnung der Differenz in Tagen und Aktualisierung des Tagesattributs
        df_sorted.loc[df_sorted[s_table_key] == user_id, 'day'] = (df_sorted.loc[df_sorted[s_table_key] == user_id, s_table_sort_by].dt.date - min_date.date()).dt.days + 1

        # Konvertieren in Ganzzahl
        df_sorted['day'] = df_sorted['day'].astype(int)

    return df_sorted


def find_path(file_name):
    # Suche den Projektordner basierend auf dem aktuellen Dateipfad
    current_dir = os.path.dirname(os.path.abspath(__file__))
    PRJ_PATH = os.path.abspath(os.path.join(current_dir, os.pardir))

    # Definiere den Pfad zum Rohdatenordner
    PRJ_DATA_RAW_PATH = os.path.join(PRJ_PATH, "data", "raw")

    # Lese die Datei im Rohdatenordner ein
    file_path = os.path.join(PRJ_DATA_RAW_PATH, file_name)
    df_map = pd.read_csv(file_path)

    return df_map


def ml_model_accuracy(X, y, model, run):
    sum_score = 0
    scores = []

    for i in range(run):
        # Aufteilung in Trainingsdaten und Testdaten
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        # Trainiere den ML-Klassifikator auf den Trainingsdaten
        model.fit(X_train, y_train)

        # Predicte das Label auf den Testdaten
        y_pred = model.predict(X_test)

        # Berechne den accuracy score durch Vergleich der predicteten Label und den tatsächlichen Labeln
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
        sum_score = sum_score + score

    print("Durchschnittliche Test Accuracy:" + str(sum_score / len(scores)))
