import pandas as pd


def get_user_timeline(df_sorted, user_id, start_day=None, end_day=None, column_name='collected_at'):
    # Eingabewerte:
    # column_name ist standardmäßig auf "collected_at" eingestellt
    # "collected_at" --> Hat der User an diesem Tag die App genutzt?
    # "value_diary_q11" --> Hat der User einen Kommentar in der App eingegeben?

    # Schritt 1: Herausfiltern aller Einträge eines spezifischen Nutzers
    user_df = df_sorted[df_sorted['user_id'] == user_id]

    # Schritt 2: Konvertieren des Datums oder Werts vom ISO-Format in Pandas-Timestamps
    user_df[column_name] = pd.to_datetime(user_df[column_name])

    # Schritt 3: Erstellen einer Liste aller Tage basierend auf den optionalen Parametern
    if start_day is None:
        start_day = min(user_df[column_name])
    if end_day is None:
        end_day = max(user_df[column_name])
    all_days = pd.date_range(start=start_day, end=end_day, freq='D').date.tolist()

    # Schritt 4: Erstellen eines binären Arrays für die User Timeline
    timeline = []
    for day in all_days:
        if day in user_df[column_name].dt.date.tolist():
            timeline.append(1)
        else:
            timeline.append(0)

    return timeline
