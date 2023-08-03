import pandas as pd
import os
from setup import *
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from util import setup

def group_and_sort(df):
    # Gruppieren des DataFrame nach user_id und Sortieren nach collected_at
    df_sorted = df.groupby(s_table_key, group_keys=True).apply(lambda x: x.sort_values([s_table_sort_by], ascending=True)).reset_index(drop=True)
    return df_sorted


def get_user_ids(df):
    # Die Methode gibt alle User IDs in einem sortierten Array aus.
    user_ids = df[s_table_key].unique()
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
    PRJ_DATA_RAW_PATH = os.path.join(PRJ_PATH, "data")

    # Lese die Datei im Rohdatenordner ein
    file_path = os.path.join(PRJ_DATA_RAW_PATH, file_name)
    df_map = pd.read_csv(file_path)

    return df_map


def data_preparation(df):
    # Löschen aller Spalten, die nur NULL-Werte enthalten
    df = df.dropna(axis='columns', how = 'all')

    # Löschen der drop_list mit allen Spalten, die nicht benötigt werden
    df = df.drop(s_table_drop_list, axis=1)

    # Umwandeln des diary Eintrags
    df[s_table_value_diary] = df[s_table_value_diary].apply(lambda x: 1 if isinstance(x, str) else 0)

    # Umwandeln von collected_at in datetime Objekte
    df[s_table_sort_by] = pd.to_datetime(df[s_table_sort_by])

    # Aufteilung des collected_at Attributs in mehrere Spalten
    df['collected_at_year'] = df[s_table_sort_by].dt.year
    df['collected_at_month'] = df[s_table_sort_by].dt.month
    df['collected_at_time'] = (df[s_table_sort_by].dt.hour * 60 + df[s_table_sort_by].dt.minute)

    # Hinzufügen des day-Attributes
    df = add_day_attribute(df)

    # Gruppieren nach s_table_key und Sortieren nach s_table_sort_by
    df = group_and_sort(df)

    return df


def filter_df_newuser(df_newuser, y):
    df_newuser_filtered = df_newuser.copy()
    df_newuser_filtered = df_newuser_filtered.drop(df_newuser_filtered[df_newuser_filtered[s_table_sort_by_alt] >= y].index)
    return df_newuser_filtered


def delete_test_user(df_sorted, new_user_id):
    if (new_user_id in get_user_ids(df_sorted)):
        return df_sorted[df_sorted['user_id'] != new_user_id]
    else:
        return df_sorted


def is_adherent(df_newuser, day):
    days = df_newuser[s_table_sort_by_alt].unique()
    if day in days.tolist():
        return 1
    else:
        return 0

