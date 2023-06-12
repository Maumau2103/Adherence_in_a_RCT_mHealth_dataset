from helper import *
from task1_phases import *
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


