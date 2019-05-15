import numpy as np
import pandas as pd



X_test = pd.read_csv('./Processed Data/history_csv_merged_with_Sire_Dam.csv', header=0)

print(X_test.groupby(['raceCourse'])['dist'].value_counts())