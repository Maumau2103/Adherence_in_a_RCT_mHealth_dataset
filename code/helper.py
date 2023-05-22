import pandas as pd

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
    user_ids = grouped_data['user_id'].unique()
    return user_ids

def add_day_attribute(df_sorted):
    # Initialisieren des Tagesattributs
    df_sorted['day'] = 0

    # Iteration über die Daten
    for user_id, group in df_sorted.groupby('user_id'):
        # Bestimmung des ältesten Datums pro user_id
        min_date = group['collected_at'].min()

        # Berechnung der Differenz in Tagen und Aktualisierung des Tagesattributs
        df_sorted.loc[df_sorted['user_id'] == user_id, 'day'] = (pd.to_datetime(df_sorted.loc[df_sorted['user_id'] == user_id, 'collected_at']) - pd.to_datetime(min_date)).dt.days + 1

        # Konvertieren in Ganzzahl
        df_sorted['day'] = df_sorted['day'].astype(int)

    return df_sorted

