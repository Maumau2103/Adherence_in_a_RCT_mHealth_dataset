import pandas as pd
import numpy as np

def get_inactive_days(group):
    inactive_days = group.collected_at.diff().dt.days - 1
    return inactive_days

def prediction_step1(df_sorted, new_user_df):
    # Schritt 1: Berechnen der Länge und Dauer der nicht-aktiven Tage für jeden Benutzer
    inactive_days = df_sorted.apply(get_inactive_days).reset_index(name='inactive_days')

    # Schritt 2: Berechnen der durchschnittlichen Länge und Dauer der nicht-aktiven Tage für jeden Benutzer
    avg_inactive_days = inactive_days.groupby("user_id")["inactive_days"].mean().reset_index(name='avg_inactive_days')

    # Schritt 3: Berechnen der durchschnittlichen Abweichung der Länge und Dauer der nicht-aktiven Tage für jeden Benutzer im Vergleich zum neuen Nutzer
    new_user_inactive_days = new_user_df.collected_at.diff().dt.days - 1
    new_user_avg_inactive_days = np.mean(new_user_inactive_days)
    avg_inactive_days["deviation"] = np.abs(avg_inactive_days["avg_inactive_days"] - new_user_avg_inactive_days)

    # Schritt 4: Sortieren der Benutzer nach aufsteigender Abweichung und Auswahl der 5 ähnlichsten Benutzer
    most_similar_users = avg_inactive_days.sort_values("deviation").head(5)["user_id"]
    print(most_similar_users)
