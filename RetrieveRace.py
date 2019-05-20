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
import pandas as pd
import re
import RaceParam as cfg

# ==================== Match Value
# date = '20190515'
# raceCourse = 'HV'
# totalMatch = '8'
totalMatch = cfg.param['totalMatch']
date = cfg.param['date']
dist = cfg.param['dist']
road = cfg.param['road']
going = cfg.param['going']
classes = cfg.param['classes']
raceCourse = cfg.param['todayRaceCourse']


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class WebCrawling:
    logging.basicConfig(filename='./Log/RetrieveRace.log', format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO)

    def __init__(self):
        self.browser = webdriver.Safari()

    def close(self):
        self.browser.quit()

    def reFormatText(self, text):
        # Trim space, remove line break, remove double space
        return re.sub(' +', ' ', text.replace('\n', ' ').replace('\r', '').replace(',', '').replace('HK$ ', '').strip())

    def getMatchInfo(self, text):
        matchInfo = text.find(
            name='table', class_='font13 lineH20 tdAlignL')

        matchInfo = matchInfo.find(name='td').text
        logging.info('Match Info: %s', matchInfo)
        regex = r"(All Weather Track|Turf, \"[A-C].*\" Course).*([0-9][0-9][0-9][0-9]M),.(.*)Prize.*(Class.[0-5])"
        matches = re.finditer(regex, matchInfo, re.MULTILINE)
        matchInfo = []
        for matchNum, match in enumerate(matches, start=1):

            # print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum = matchNum, start = match.start(), end = match.end(), match = match.group()))

            for groupNum in range(0, len(match.groups())):
                groupNum = groupNum + 1
                matchInfo.append(match.group(groupNum))
                # print ("Group {groupNum} found at {start}-{end}: {group}".format(groupNum = groupNum, start = match.start(groupNum), end = match.end(groupNum), group = match.group(groupNum)))
        return matchInfo

    def getMatchDetail(self, text):
        table = text.find(name='table', class_='draggable hiddenable')
        MatchDetail = []
        for tr in table.find_all(name='tr'):
            for td in tr.find_all(name='td'):
                MatchDetail.append(self.reFormatText(td.text))

        MatchDetail = numpy.reshape(MatchDetail, (-1, 25))
        MatchDetail = pd.DataFrame(data=MatchDetail[1:, 0:],
                                   columns=MatchDetail[0, 0:])
        return MatchDetail

    def request(self, date, raceCourse, matchNo):
        self.url = 'https://racing.hkjc.com/racing/Info/Meeting/RaceCard/English/Local/' + \
            date+'/'+raceCourse+'/'+matchNo
        self.browser.get(self.url)
        try:

            page_source = WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[@id='racecard']/div[8]/table")))

            html_source = bs4.BeautifulSoup(self.browser.page_source, 'lxml')
            return html_source

        except TimeoutException as ex:
            logging.error('Loading took too much time!')
            logging.exception(ex)
            logging.error('URL with %s', self.url)
            return


# ============== WebCrawling (Start) ============
allMatchToday = pd.DataFrame()

try:
    # =============== initial webCrawling
    logging.info('Initital Web Crawling - Launch Selenium Browser')
    webCrawling = WebCrawling()

    for matchNo in range(1, int(totalMatch) + 1):

        # ============== retrive full page source
        html_source = webCrawling.request(str(date), raceCourse, str(matchNo))

        # ============== retrive match base information
        matchInfo = webCrawling.getMatchInfo(html_source)
        logging.info('Match Info: %s %s', np.shape(matchInfo), matchInfo)

        # ============== retrive match detail information
        matchDetail = webCrawling.getMatchDetail(html_source)
        logging.info('Match Detail: \n %s', matchDetail)

        # ========== drop useless data
        matchDetail = matchDetail[(matchDetail['Jockey'] != '-')]
        logging.info('Match Detail: \n %s', matchDetail)

        # ============== Split jockey awt in Match Detail
        split_result = matchDetail['Jockey'].astype(
            str).str.split("(", n=1, expand=True)
        matchDetail['Jockey'] = split_result[0]
        logging.info('split_result shape : %s', str(np.shape(split_result)))
        if ', 2)' in str(np.shape(split_result)):
            logging.info('In if')
            matchDetail['AWT'] = split_result[1].str.replace(')', '')
            matchDetail['AWT'].fillna(value=0, inplace=True)
            matchDetail['AWT'] = matchDetail['Wt.'].astype(
                float) + matchDetail['AWT'].astype(float)
        else:
            logging.info('In else')
            matchDetail['AWT'] = matchDetail['Wt.']

        matchDetail = matchDetail.drop(['Wt.'], axis=1)

        # ============== Merge match information to match detail
        matchDetail['road'] = matchInfo[0]
        matchDetail['dist'] = matchInfo[1]
        matchDetail['going'] = matchInfo[2]
        matchDetail['class'] = matchInfo[3]
        matchDetail['raceNo'] = matchNo
        matchDetail['raceCourse'] = raceCourse
        logging.info('Match Detail: %s \n %s',
                     np.shape(matchDetail), matchDetail)

        # ============== Add to all match today
        allMatchToday = allMatchToday.append(matchDetail)

    logging.info('All Match Today: %s \n %s',
                 np.shape(allMatchToday), allMatchToday)

except Exception as ex:
    logging.exception(ex)

finally:
    logging.info('Close Selenium Browser')
    webCrawling.close()

# ============== WebCrawling (End) =============


# ============== Editing data for prediction  (Start) =================

# ============== Read required csv
# sireRank = pd.read_csv('./Processed Data/sireRank.csv', sep=',')
# damRank = pd.read_csv('./Processed Data/damRank.csv', sep=',')
# jockey = pd.read_csv('./Processed Data/jockeyRank.csv', sep=',')
# trainer = pd.read_csv('./Processed Data/trainerRank.csv', sep=',')


# ============== Join required csv with allMatchToday
# allMatchToday = pd.merge(allMatchToday, sireRank, how='left',
#                          left_on=['Sire'], right_on=['Sire'])

# allMatchToday = pd.merge(allMatchToday, damRank, how='left',
#                          left_on=['Dam'], right_on=['Dam'])

# allMatchToday = pd.merge(allMatchToday, jockey, how='left',
#                          left_on=['Jockey'], right_on=['Jockey'])

# allMatchToday = pd.merge(allMatchToday, trainer, how='left',
#                          left_on=['Trainer'], right_on=['Trainer'])


# ============= Update road and going
allMatchToday['road'] = allMatchToday['road'].str.replace(
    ',', ' -').str.replace('"', '').str.upper().str.replace('COURSE', 'Course')
allMatchToday['going'] = allMatchToday['going'].str.upper()

logging.info('All Match Today: %s \n %s',
             np.shape(allMatchToday), allMatchToday)


# ============== Editing data for prediction  (End) =================


# ================ Save as csv ==================
headers = ','.join(map(str, allMatchToday.columns.values))
np.savetxt('./Processed Data/match_data_'+date+'.csv', allMatchToday.round(0),
           delimiter=',', fmt='%s', header=headers, comments='')
logging.info('Finished write csv')
