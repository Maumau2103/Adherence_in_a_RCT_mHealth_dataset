from task4_statistics import *

df_sorted = pd.read_csv('C:/Users/mauri/PycharmProjects/Softwareprojekt/data/dataset_sorted.csv', parse_dates=['collected_at'])

statistics(df_sorted, 2107)
