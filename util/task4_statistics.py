from helper import *
from setup import *
from task1_phases import *
from task2_groups import *
from task3_prediction import *
from task5_adherence_level import *
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt


def show_user_timeline(df_sorted, user_id, step=50):
    # Berechnen der Timeline für einen User
    timeline = get_user_timeline(df_sorted, user_id)

    # X-Achse: Indizes der Elemente im Array
    x = np.arange(len(timeline))

    # Figure-Objekt erstellen und Größe festlegen
    fig, ax = plt.subplots(figsize=(12, 4))

    # Scatter Plot erstellen
    ax.scatter(x, np.zeros_like(timeline), c=timeline, cmap='binary', marker='o')

    # Achsentitel und Diagrammtitel festlegen
    ax.set_xlabel('Days')
    ax.set_ylabel('Adherence')
    ax.set_title('Users Timeline')

    # Ersten und letzten Wert für die x-Achse festlegen
    x_start = 0
    x_end = (len(x) - 1) // 10 * 10

    # x-Achse beschriften
    x_ticks = np.arange(x_start, x_end + 1, step)
    x_labels = x_ticks
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    # Achsenbeschriftungen ausblenden
    ax.set_yticks([])

    # Legende hinzufügen
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='lightgrey', markerfacecolor='k', markersize=10, label='adherent'),
        plt.Line2D([0], [0], marker='o', color='lightgrey', markerfacecolor='w', markersize=10, label='not adherent')]
    plt.legend(handles=legend_elements, facecolor='lightgrey')

    # Diagramm anzeigen
    plt.show()


def show_user_statistics(df_sorted, user_id):
    # Filtere den Datensatz für den gegebenen Nutzer
    user_data = df_sorted[df_sorted[s_table_key] == user_id]

    # users timeline
    timeline = get_user_timeline(user_data, user_id)

    if user_data.empty:
        print('user not found')
        return 0  # Wenn der Nutzer nicht im DataFrame gefunden wurde, gib 0 zurück

    # Sortiere den DataFrame nach der Spalte "day"
    user_data_sorted = user_data.sort_values(by='day').reset_index(drop=True)

    # Finde den längsten aufeinanderfolgenden Streak von "day"
    current_streak = 1
    max_streak = 1

    for i in range(1, len(user_data_sorted)):
        if user_data_sorted.iloc[i]['day'] <= user_data_sorted.iloc[i-1]['day'] + 1:
            current_streak += 1
        else:
            current_streak = 1

        max_streak = max(max_streak, current_streak)

    # Ermittle die Anzahl der Lücken in der Spalte "day"
    anzahl_luecken = 0

    for i in range(1, len(user_data_sorted)):
        diff = user_data_sorted.iloc[i]['day'] - user_data_sorted.iloc[i-1]['day']
        if diff > 1:
            anzahl_luecken += 1

    # Berechne statistische Daten für den Nutzer
    user_statistics = {
        'user_id': user_id,
        'Anzahl_Tage': len(timeline),
        'Anzahl_Einträge': timeline.count(1),
        'Anzahl_fehlende_Einträge': timeline.count(0),
        'längster_adh_Streak': max_streak,
        'adh_percentage': get_user_adh_percentage(user_data, user_id),
        'Anzahl_Lücken': anzahl_luecken
    }

    # Erstelle ein DataFrame mit den statistischen Daten des Nutzers
    result_df = pd.DataFrame([user_statistics])

    return result_df


def show_cluster_timelines(df_sorted):
    cluster = cluster_timelines(df_sorted)


def show_user_statistics_2(df, user_id, step=50):
    new_df = df[df['user_id'] == user_id]
    df_raw = data_preparation(new_df)
    filtered_df = df_raw.sort_values('day')
    max_day_value = filtered_df['day'].max()

    id = filtered_df['id']
    day = filtered_df['day']
    locale = filtered_df['locale']
    client = filtered_df['client']
    collected_at = filtered_df['collected_at']
    value_loudness = filtered_df['value_loudness']
    value_cumberness = filtered_df['value_jawbone']
    value_jawbone = filtered_df['value_jawbone']
    value_neck = filtered_df['value_neck']
    value_tin_day = filtered_df['value_tin_day']
    value_tin_cumber = filtered_df['value_tin_cumber']
    value_tin_max = filtered_df['value_tin_max']
    value_movement = filtered_df['value_movement']
    value_stress = filtered_df['value_stress']
    value_emotion = filtered_df['value_emotion']
    value_diary_q11 = filtered_df['value_diary_q11']

    df_dict = {"loudness": value_loudness,
                "cumberness": value_cumberness,
                "jawbone": value_jawbone,
                "neck": value_neck,
                "tin_day": value_tin_day,
                "tin_cumber": value_tin_cumber,
                "tin_max": value_tin_max,
                "movement": value_movement,
                "stress": value_stress,
                "emotion": value_emotion,
               }

    show_df = pd.DataFrame(df_dict, index=day)
    show_df.plot()
    plt.show()


def show_user_statistics3(df, user_id, step=10):
    new_df = df[df['user_id'] == user_id]
    df_raw = data_preparation(new_df)
    filtered_df = df_raw.sort_values('day')
    max_day_value = filtered_df['day'].max()

    selected_columns = ['day', 'value_loudness', 'value_cumberness', 'value_jawbone', 'value_neck', 'value_tin_day',
                        'value_tin_cumber', 'value_tin_max', 'value_movement', 'value_stress', 'value_emotion']
    show_df = filtered_df[selected_columns]

    show_df = show_df.iloc[::step]  # Nur jeden step-ten Wert auswählen

    show_df.plot(x='day')
    plt.show()

