from helper import *
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt


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


def get_all_user_timelines(df_sorted, start_day=None, end_day=None, column_name='collected_at'):
    # Diese Methode soll alle User Timelines in einem mehrdimensionalen Array ausgeben.
    # Eingabewerte: --> siehe Methode "get_user_timeline"

    user_ids = get_user_ids(df_sorted)
    timelines = []

    for user_id in user_ids:
        timeline = get_user_timeline(df_sorted, user_id, start_day=start_day, end_day=end_day, column_name=column_name)
        timelines.append(timeline)

    return timelines


def calculate_percentage(timelines):
    num_users = len(timelines)
    timeline_length = len(timelines[0])
    adherence_percentages = [0] * timeline_length

    for timeline in timelines:
        for i in range(timeline_length):
            if timeline[i] == 1:
                adherence_percentages[i] += 1

    for i in range(timeline_length):
        adherence_percentages[i] = (adherence_percentages[i] / num_users) * 100

    return adherence_percentages

def cpd_binseg(adherence_percentages):

    # Ruptures liefert mehrere spezifische Anwendungfsfälle, wie z.B. die Erkennung von Mustern.
    # Da wir hier einen Anwendungfall haben in dem die prozentuale Nutzung der User exponentiell abnimmt, nutzen wir
    # das Modell "exponential"

    model = "exponential"

    # Ruptures enthält eine Methode für die Binäre Segmentierung. Wie oben beschrieben geben wir bei dem Parameter
    # "exponentiell" ein. Die Methode fit() wird genutzt um verschiedene Modellparameter nutzen zu können. Viel mehr
    # Wissen über diese Methode ist an dieser Stelle nicht nötig um den Code zu verstehen.

    algo = rpt.Binseg(model=model).fit(adherence_percentages)