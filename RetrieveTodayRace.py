from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

import bs4
import logging
import time
import re
import numpy
import numpy as np
import pandas
import re

date = '20190511'
raceCource = 'ST'
totalMatch = '8'

url = 'https://racing.hkjc.com/racing/Info/Meeting/RaceCard/English/Local/' + \
    date+'/'+raceCource+'/'+totalMatch


browser = webdriver.Safari()
browser.get(url)


def reFormatText(text):
    # Trim space, remove line break, remove double space
    return re.sub(' +', ' ', text.replace('\n', ' ').replace('\r', '').replace(',', '').replace('HK$ ', '').strip())


try:

    page_source = WebDriverWait(browser, 10).until(
        EC.presence_of_element_located((By.XPATH, "//*[@id='racecard']/div[8]/table")))

    html_source = bs4.BeautifulSoup(browser.page_source, 'lxml')
# print(html_source)

except TimeoutException as ex:
    print('error')
    # logging.error('Loading took too much time!')
    # logging.error('Exception found %s', ex)
    # logging.error('URL with %s', url)
    browser.quit()
browser.quit()


matchInfo = html_source.find(
    name='table', class_='font13 lineH20 tdAlignL')

matchInfo = matchInfo.find(name='td').text
regex = r"(All Weather Track|Turf, \"[A-C].*\" Course).*([0-9][0-9][0-9][0-9]M),.(.*)Prize.*(Class.[0-5])"
matches = re.finditer(regex, matchInfo, re.MULTILINE)
matchInfo = []
for matchNum, match in enumerate(matches, start=1):

    # print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum = matchNum, start = match.start(), end = match.end(), match = match.group()))

    for groupNum in range(0, len(match.groups())):
        groupNum = groupNum + 1
        matchInfo.append(match.group(groupNum))
        # print ("Group {groupNum} found at {start}-{end}: {group}".format(groupNum = groupNum, start = match.start(groupNum), end = match.end(groupNum), group = match.group(groupNum)))
print(matchInfo)
classes = matchInfo[2]
classes = matchInfo[2]


table = html_source.find(
    name='table', class_='draggable hiddenable')


MatchDetail = []
for tr in table.find_all(name='tr'):
    MatchDetailTemp = []
    for td in tr.find_all(name='td'):
        MatchDetailTemp.append(reFormatText(td.text))
    MatchDetailTemp += matchInfo
    MatchDetail.append(MatchDetailTemp)
# print(MatchDetail)
MatchDetail = numpy.reshape(MatchDetail, (-1, 29))

# print(MatchDetail)
MatchDetail = numpy.delete(MatchDetail, (0), axis=0)
headers = 'Horse No.,Last 6 Runs,Colour,Horse,Brand No.,Wt.,Jockey,Over Wt.,Draw,Trainer,Rtg.,Rtg.+/-,Horse Wt. (Declaration),Wt.+/- (vs Declaration),Best Time,Age,WFA,Sex,Season Stakes,Priority,Gear,Owner,Sire,Dam,Import Cat.,road,dist,going,class'
numpy.savetxt('todayMatch.csv', MatchDetail, delimiter=',',
              fmt='%s', comments='', header=headers)

# exit()
todayMatch = pandas.read_csv('todayMatch.csv', sep=',')

# print(todayMatch['Jockey'].head())
# print(todayMatch['Wt.'].head())
split_result = todayMatch['Jockey'].astype(str).str.split("(", n=1, expand=True)
print(split_result)
todayMatch['Jockey'] = split_result[0]
todayMatch['deductAWT'] = split_result[1].str.replace(
    ')', '')
todayMatch['deductAWT'].fillna(value=0, inplace=True)
# print(todayMatch['deductAWT'])

todayMatch['Wt.'] = todayMatch['Wt.'].astype(
    float) + todayMatch['deductAWT'].astype(float)

# print(todayMatch['Jockey'].head())
# print(todayMatch['Wt.'].head())

headers = ','.join(map(str, todayMatch.columns.values))
numpy.savetxt('todayMatch.csv', todayMatch, delimiter=',',
              fmt='%s',  header=headers, comments='')


todayMatch = pandas.read_csv('todayMatch.csv', sep=',')
sireRank = pandas.read_csv('./Processed Data/sireRank.csv', sep=',')
damRank = pandas.read_csv('./Processed Data/damRank.csv', sep=',')
damRank = pandas.read_csv('./Processed Data/damRank.csv', sep=',')
jockey1819 = pandas.read_csv('./Raw Data/jockey1819.csv', sep=',')
jockey1819['Jockey'] = jockey1819['Jockey'].str.strip()
trainer1819 = pandas.read_csv('./Raw Data/trainer1819.csv', sep=',')

todayMatch = pandas.merge(todayMatch, sireRank, how='left',
                          left_on=['Sire'], right_on=['Sire'])

todayMatch = pandas.merge(todayMatch, damRank, how='left',
                          left_on=['Dam'], right_on=['Dam'])

todayMatch = pandas.merge(todayMatch, jockey1819, how='left',
                          left_on=['Jockey'], right_on=['Jockey'])

todayMatch = pandas.merge(todayMatch, trainer1819, how='left',
                          left_on=['Trainer'], right_on=['Trainer'])

print(','.join(map(str, todayMatch.columns.values)))

todayMatch = todayMatch[['Horse No.', 'Horse', 'Brand No.',
                         'Wt.', 'Draw', 'Horse Wt. (Declaration)', 'Age', 'SireRank', 'DamRank', 'Win_x', 'Win_y', 'road', 'dist', 'going', 'class']]

# todayMatch['class'] = classes
todayMatch['road'] = todayMatch['road'].str.upper()
# todayMatch['dist'] = dist
# todayMatch['going'] = going
todayMatch['raceCource'] = raceCource

todayMatch = pandas.get_dummies(
    todayMatch, columns=['class'], prefix=['class'])

todayMatch = pandas.get_dummies(
    todayMatch, columns=['road'], prefix=['road'])

todayMatch = pandas.get_dummies(
    todayMatch, columns=['dist'], prefix=['dist'])

todayMatch = pandas.get_dummies(
    todayMatch, columns=['going'], prefix=['going'])

todayMatch = pandas.get_dummies(
    todayMatch, columns=['raceCource'], prefix=['raceCource'])

print(','.join(map(str, todayMatch.columns.values)))

if 'dist_1000M' not in todayMatch:
    todayMatch['dist_1000M'] = np.NaN

if 'dist_1200M' not in todayMatch:
    todayMatch['dist_1200M'] = np.NaN

if 'dist_1000M' not in todayMatch:
    todayMatch['dist_1000M'] = np.NaN

if 'dist_1400M' not in todayMatch:
    todayMatch['dist_1400M'] = np.NaN

if 'dist_1600M' not in todayMatch:
    todayMatch['dist_1600M'] = np.NaN

if 'dist_1650M' not in todayMatch:
    todayMatch['dist_1650M'] = np.NaN

if 'dist_1800M' not in todayMatch:
    todayMatch['dist_1800M'] = np.NaN

if 'dist_2000M' not in todayMatch:
    todayMatch['dist_2000M'] = np.NaN

if 'road_ALL WEATHER TRACK' not in todayMatch:
    todayMatch['road_ALL WEATHER TRACK'] = np.NaN

if 'road_TURF - A Course' not in todayMatch:
    todayMatch['road_TURF - A Course'] = np.NaN

if 'road_TURF - B Course' not in todayMatch:
    todayMatch['road_TURF - B Course'] = np.NaN

if 'road_TURF - C Course' not in todayMatch:
    todayMatch['road_TURF - C Course'] = np.NaN

if 'going_GOOD' not in todayMatch:
    todayMatch['going_GOOD'] = np.NaN

if 'going_GOOD TO FIRM' not in todayMatch:
    todayMatch['going_GOOD TO FIRM'] = np.NaN

if 'rgoing_GOOD TO YIELDING' not in todayMatch:
    todayMatch['going_GOOD TO YIELDING'] = np.NaN

if 'class_Class 1' not in todayMatch:
    todayMatch['class_Class 1'] = np.NaN

if 'class_Class 2' not in todayMatch:
    todayMatch['class_Class 2'] = np.NaN

if 'class_Class 3' not in todayMatch:
    todayMatch['class_Class 3'] = np.NaN

if 'class_Class 4' not in todayMatch:
    todayMatch['class_Class 4'] = np.NaN

if 'class_Class 5' not in todayMatch:
    todayMatch['class_Class 5'] = np.NaN

if 'raceCource_HV' not in todayMatch:
    todayMatch['raceCource_HV'] = np.NaN

if 'raceCource_ST' not in todayMatch:
    todayMatch['raceCource_ST'] = np.NaN


todayMatch = todayMatch.rename(
    columns={'Win_y': 'Win%_y',
             'Win_x': 'Win%_x',
             'Wt.': 'awt',
             'Draw': 'draw',
             'Horse Wt. (Declaration)': 'dhw',
             })

todayMatch['Win%_y'] = todayMatch['Win%_y'] * 100
todayMatch['Win%_x'] = todayMatch['Win%_x'] * 100
todayMatch = todayMatch[['Horse No.','Horse' ,'Brand No.','draw','Age','Win%_y','Win%_x','DamRank','SireRank','awt','dhw','dist_1000M','dist_1200M','dist_1400M','dist_1600M','dist_1650M','dist_1800M','dist_2000M','road_ALL WEATHER TRACK','road_TURF - A Course','road_TURF - B Course','road_TURF - C Course','going_GOOD','going_GOOD TO FIRM','going_GOOD TO YIELDING','class_Class 1','class_Class 2','class_Class 3','class_Class 4','class_Class 5','raceCource_HV','raceCource_ST']]
print(','.join(map(str, todayMatch.columns.values)))
print(todayMatch)
headers = ','.join(map(str, todayMatch.columns.values))
numpy.savetxt('todayPredictValue.csv', todayMatch, delimiter=',',
              fmt='%s',  header=headers, comments='')
