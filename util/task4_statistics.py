from helper import *
from setup import *
from task1_phases import *
from task2_groups import *
from task3_prediction import *
from task5_adherence_level import *
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import seaborn as sns


def show_user_timeline(df_sorted, user_id, result_phases, start_day=None, end_day=None, step=50):
    # Berechnen der Timeline für einen User
    if user_id in df_sorted[s_table_key].tolist():
        timeline = get_user_timeline(df_sorted, user_id, start_day, end_day)
    else:
        print('user not found')
        return 0

    # X-Achse: Indizes der Elemente im Array
    x = np.arange(len(timeline))

    # Figure-Objekt erstellen und Größe festlegen
    fig, ax = plt.subplots(figsize=(12, 4))

    # Hintergrundfarbe auf hellgrau setzen
    ax.set_facecolor('lightgrey')

    # Scatter Plot erstellen
    ax.scatter(x, np.zeros_like(timeline), c=timeline, cmap='binary', marker='o')

    # Achsentitel und Diagrammtitel festlegen
    ax.set_xlabel('Days')
    ax.set_ylabel('Adherence')
    ax.set_title('Users Timeline')

    # Ersten und letzten Wert für die x-Achse festlegen
    x_start = 0
    x_end = (len(x)) // 10 * 10

    # Vertikale Linie zur Abgrenzung der Phasen anzeigen
    for change_point in result_phases:
        ax.axvline(x=change_point, color='black', linestyle='dashed', alpha=0.7)

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
        plt.Line2D([0], [0], marker='o', color='lightgrey', markerfacecolor='w', markersize=10, label='not adherent'),
        plt.Line2D([0], [0], color='black', linestyle='dashed', alpha=0.7, label='Change Point')]
    plt.legend(handles=legend_elements, facecolor='white')

    # Diagramm anzeigen
    plt.show()


def show_users_clusters(clusters):
    # Erstellen von Features und Labels für die Reduzierung
    X = clusters.iloc[:, 1].tolist()
    y = clusters['cluster_label']

    # Reduzieren der Datenzeilen auf 2 Features
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Speichere die neuen Features in einem neuen DataFrame ab
    df_2d = pd.DataFrame(X_2d, columns=['Feature_1', 'Feature_2'])
    df_2d['cluster_label'] = y

    # Erstellung einer benutzerdefinierten Farbpalette basierend auf der Anzahl der eindeutigen Cluster
    unique_clusters = df_2d['cluster_label'].unique()
    colors = sns.color_palette('husl', n_colors=len(unique_clusters))

    # Erstellen der Figure und des Koordinatensystems
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Schleife durch die eindeutigen Cluster
    for i, cluster in enumerate(unique_clusters):
        # Filtern der Datenzeilen für den aktuellen Cluster
        cluster_data = df_2d[df_2d['cluster_label'] == cluster]

        # Plotten der Punkte für den aktuellen Cluster in der entsprechenden Farbe
        ax.scatter(cluster_data['Feature_1'], cluster_data['Feature_2'], c=[colors[i]], label=f'Cluster {cluster}',
                   s=100)

    # Beschriftungen für die Achsen und die Legende
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.title('result of the clustering')
    plt.legend(title='Cluster', loc='upper right')

    # Anzeigen der Grafik
    plt.show()

    return df_2d


def show_user_adherence_percentage(users_phases, user_id=None):
    if user_id is None:
        user_adherence_percentages = users_phases
    else:
        user_phases = users_phases[users_phases[s_table_key] == user_id]
        user_adherence_percentages = user_phases.iloc[0]['phases']

    print(user_adherence_percentages)

    # Erstelle eine Liste mit Phasen-Indizes
    phases = list(range(1, len(user_adherence_percentages) + 1))

    # Setze den Stil von Seaborn
    sns.set(style="white")
    muted_palette = sns.color_palette("muted")

    # Erstelle das Plot-Diagramm als Balkendiagramm
    plt.figure(figsize=(7, 5))
    plt.bar(phases, user_adherence_percentages, color=muted_palette, width=0.7)

    # Beschrifte die Achsen und den Titel des Diagramms
    plt.xlabel('Phases')
    plt.ylabel('Adherence percentages')
    plt.title('Adherence in verschiedenen Phasen')

    # Setze die Beschriftung der x-Achse auf "Phase 1", "Phase 2", usw.
    plt.xticks(phases, ['Phase ' + str(phase) for phase in phases])

    # Zeige das Diagramm
    plt.show()


def show_user_statistics(df_sorted, user_id):
    # Filtere den Datensatz für den gegebenen Nutzer
    user_data = df_sorted[df_sorted[s_table_key] == user_id]

    if user_data.empty:
        print('user not found')
        return 0  # Wenn der Nutzer nicht im DataFrame gefunden wurde, gib 0 zurück

    # users timeline
    timeline = get_user_timeline(user_data, user_id, start_day=None, end_day=None)

    # Finde den längsten aufeinanderfolgenden Streak von "day"
    current_streak = 1
    max_streak = 1

    for i in range(1, len(user_data)):
        if user_data.iloc[i][s_table_sort_by_alt] <= user_data.iloc[i-1][s_table_sort_by_alt] + 1:
            current_streak += 1
        else:
            current_streak = 1

        max_streak = max(max_streak, current_streak)

    # Ermittle die Anzahl der Lücken in der Spalte "day"
    anzahl_luecken = 0

    for i in range(1, len(user_data)):
        diff = user_data.iloc[i][s_table_sort_by_alt] - user_data.iloc[i-1][s_table_sort_by_alt]
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