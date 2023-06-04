import pandas as pd
import os

def csv_to_dataframe(file_path):
    """
    Read a CSV file and return a DataFrame object.
    :param file_path: The path to the CSV file to read.
    :return: A DataFrame object representing the data in the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")


def group_and_sort(df):

    # Gruppieren des DataFrame nach user_id und Sortieren nach collected_at

    df_sorted = df.groupby("user_id").apply(lambda x: x.sort_values(["collected_at"], ascending=True)).reset_index(drop=True)
    return df_sorted


def get_user_ids(grouped_data):
    # Die Methode gibt alle User IDs in einem sortierten Array aus.
    user_ids = grouped_data['user_id'].unique()
    return user_ids


def add_day_attribute(df_sorted):
    # Konvertieren von 'collected_at' in das Datumsformat
    df_sorted['collected_at'] = pd.to_datetime(df_sorted['collected_at'])

    # Initialisieren des Tagesattributs
    df_sorted['day'] = 0

    # Iteration über die Daten
    for user_id, group in df_sorted.groupby('user_id'):
        # Bestimmung des ältesten Datums pro user_id
        min_date = group['collected_at'].min()

        # Berechnung der Differenz in Tagen und Aktualisierung des Tagesattributs
        df_sorted.loc[df_sorted['user_id'] == user_id, 'day'] = (df_sorted.loc[df_sorted['user_id'] == user_id, 'collected_at'].dt.date - min_date.date()).dt.days + 1

        # Konvertieren in Ganzzahl
        df_sorted['day'] = df_sorted['day'].astype(int)

    return df_sorted


def get_all_days(df_newuser):
    # Überprüfen, ob die Spalte "day" vorhanden ist
    if 'day' not in df_newuser.columns:
        raise ValueError("Die Spalte 'day' existiert nicht im Datensatz.")

    # Eindeutige Werte für das Attribut "day" in einer Liste speichern
    all_days = df_newuser['day'].unique().tolist()

    return all_days


def filter_by_day(df_similarusers, day):
    # Überprüfen, ob die Spalte "day" vorhanden ist
    if 'day' not in df_similarusers.columns:
        raise ValueError("Die Spalte 'day' existiert nicht im Datensatz.")

    # Filtern der Zeilen mit dem angegebenen Wert für "day"
    filtered_data = df_similarusers[df_similarusers['day'] == day]

    return filtered_data

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
