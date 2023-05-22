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
    df_sorted = df.groupby("user_id").apply(lambda x: x.sort_values(["collected_at"], ascending=True)).reset_index(
        drop=True)

    # Hinzufügen des Attributs day
    # Initialisieren des Tagesattributs
    df_sorted['day'] = 0

    # Iteration über die Daten
    for user_id, group in df_sorted.groupby('user_id'):
        day_counter = 1

        for index, row in group.iterrows():
            # Aktualisieren des Tagesattributs für vorhandene Daten
            df_sorted.at[index, 'day'] = day_counter
            day_counter += 1

    return df_sorted



