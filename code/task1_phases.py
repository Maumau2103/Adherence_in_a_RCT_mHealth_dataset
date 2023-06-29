from helper import *
import ruptures as rpt
from setup import *
def get_user_timeline(df_sorted, key_column, start_day=None, end_day=None, column=s_table_sort_by):
    # Eingabewerte:
    # column_name ist standardmäßig auf "collected_at" eingestellt
    # "collected_at" --> Hat der User an diesem Tag die App genutzt?
    # "value_diary_q11" --> Hat der User einen Kommentar in der App eingegeben?

    # Schritt 1: Herausfiltern aller Einträge eines spezifischen Nutzers
    user_df = df_sorted[df_sorted[s_table_key] == key_column]

    # Schritt 2: Konvertieren des Datums oder Werts vom ISO-Format in Pandas-Timestamps
    user_df = user_df.copy()
    user_df.loc[:, column] = pd.to_datetime(user_df[column])

    # Schritt 3: Erstellen einer Liste aller Tage basierend auf den optionalen Parametern
    if start_day is None:
        start_day = min(user_df[column])
    if end_day is None:
        end_day = max(user_df[column])
    all_days = pd.date_range(start=start_day, end=end_day, freq='D').date.tolist()

    # Schritt 4: Erstellen eines binären Arrays für die User Timeline
    timeline = []
    for day in all_days:
        if day in user_df[column].dt.date.tolist():
            timeline.append(1)
        else:
            timeline.append(0)

    return timeline


def get_all_user_timelines(df_sorted, start_day=None, end_day=None):
    # Diese Methode soll alle User Timelines in einem mehrdimensionalen Array ausgeben.
    # Eingabewerte: --> siehe Methode "get_user_timeline"

    user_ids = get_user_ids(df_sorted)
    timelines = []

    for i in range(len(user_ids)):
        timeline = get_user_timeline(df_sorted, user_ids[i], start_day=start_day, end_day=end_day, column=s_table_sort_by)
        timelines.append(timeline)

    return timelines


def get_all_adherence_percentage(timelines):
    num_users = len(timelines)
    timeline_length = len(timelines[0])
    all_adherence_percentages = [0] * timeline_length

    for timeline in timelines:
        for i in range(timeline_length):
            if timeline[i] == 1:
                all_adherence_percentages[i] += 1

    for i in range(timeline_length):
        all_adherence_percentages[i] = (all_adherence_percentages[i] / num_users) * 100

    return all_adherence_percentages


def cpd_binseg(all_adherence_percentages):
    # Ruptures liefert mehrere spezifische Anwendungfsfälle, wie z.B. die Erkennung von Mustern.
    # Da wir hier einen Anwendungfall haben in dem die prozentuale Nutzung der User exponentiell abnimmt, nutzen wir
    # das Modell "exponential"

    model = "exponential"

    # signal = all_adherence_percentages

    # Ruptures enthält eine Methode für die Binäre Segmentierung. Wie oben beschrieben geben wir bei dem Parameter
    # "exponentiell" ein. Die Methode fit() wird genutzt um verschiedene Modellparameter nutzen zu können. Viel mehr
    # Wissen über diese Methode ist an dieser Stelle nicht nötig um den Code zu verstehen.

    algo = rpt.Binseg(model=model).fit(all_adherence_percentages)

    my_bkps = algo.predict(n_bkps=3)

    my_bkps = algo.predict(pen=np.log(n) * dim * sigma ** 2)
    # or
    # my_bkps = algo.predict(epsilon=3 * n * sigma ** 2)

    # rpt.show.display(signal, bkps, my_bkps, figsize=(10, 6))
    # plt.show()
    return my_bkps

def cpd_botupseg(all_adherence_percentages):

    model = "exponential"

    algo = rpt.BottomUp(model=model).fit(all_adherence_percentages)
    my_bkps = algo.predict(n_bkps=3)

    return my_bkps

def cpd_windowseg(all_adherence_percentages):

    model = "exponential"

    algo = rpt.Window(width=40, model=model).fit(all_adherence_percentages)
    my_bkps = algo.predict(n_bkps=3)

    return my_bkps
