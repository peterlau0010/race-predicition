import pandas as pd

date = '20190518'
totalMatch = '10'
todayRaceCourse = 'ST'


data = pd.read_csv('./Processed Data/match_data_'+date+'.csv', header=0)
data = data[['raceCourse', 'class', 'dist', 'road', 'going', 'raceNo']]
data = data.drop_duplicates()
# print(data)

param = {
    'raceCourse': data['raceCourse'].tolist(),
    'classes': data['class'].tolist(),
    'dist': data['dist'].tolist(),
    'road': 'TURF',
    'going': data['going'].tolist(),
    'date': date,
    'totalMatch': totalMatch,
    'todayRaceCourse': todayRaceCourse
}

# print(param)
