
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


# ==================== Match Value
jockey = 1


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class WebCrawling:
    logging.basicConfig(filename='./Log/RetrieveJockeyTrainer.log', format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S ', level=logging.INFO)

    def __init__(self):
        self.browser = webdriver.Safari()

    def close(self):
        self.browser.quit()

    def reFormatText(self, text):
        # Trim space, remove line break, remove double space
        return re.sub(' +', ' ', text.replace('\n', ' ').replace('\r', '').replace(',', '').replace('HK$ ', '').strip())

    def getListDetail(self, text):
        table = text.find(name='table', class_='table_bd')
        listDetail = []
        for tr in table.find_all(name='tr'):
            for td in tr.find_all(name='td'):
                listDetail.append(self.reFormatText(td.text))

        listDetail.remove('Others')
        if self.jockey > 0:
            listDetail.remove("Jockeys' Ranking")
            listDetail.remove('Jockeys in Service')
        else:
            listDetail.remove("Trainers' Ranking")
            listDetail.remove('Trainers in Service')

        listDetail.remove('View in Numbers')
        listDetail.remove('')
        listDetail = numpy.reshape(listDetail, (-1, 8))
        listDetail = pd.DataFrame(data=listDetail[1:, 0:],
                                  columns=listDetail[0, 0:])
        return listDetail

    def request(self, jockey):
        self.jockey = jockey
        if jockey > 0:
            self.url = 'https://racing.hkjc.com/racing/information/english/Jockey/JockeyRanking.aspx?Season=Current&View=Percentage&Racecourse=ALL'
        else:
            self.url = 'https://racing.hkjc.com/racing/information/English/Trainers/TrainerRanking.aspx?Season=Current&View=Percentage&Racecourse=ALL'
        self.browser.get(self.url)
        try:

            page_source = WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.XPATH, "/html/body/div/div[2]/div[2]/table")))

            html_source = bs4.BeautifulSoup(self.browser.page_source, 'lxml')
            return html_source

        except TimeoutException as ex:
            logging.error('Loading took too much time!')
            logging.exception(ex)
            logging.error('URL with %s', self.url)
            return


try:
    # =============== initial webCrawling
    logging.info('Initital Web Crawling - Launch Selenium Browser')
    webCrawling = WebCrawling()

    html_source = webCrawling.request(jockey)
    jockeytrainerlist = webCrawling.getListDetail(html_source)
    jockeytrainerlist['Win'] = jockeytrainerlist['Win'].str.strip('%')
    jockeytrainerlist['2nd'] = jockeytrainerlist['2nd'].str.strip('%')
    jockeytrainerlist['3rd'] = jockeytrainerlist['3rd'].str.strip('%')
    jockeytrainerlist['4th'] = jockeytrainerlist['4th'].str.strip('%')
    jockeytrainerlist['5th'] = jockeytrainerlist['5th'].str.strip('%')
    jockeytrainerlist['Stakes Won'] = jockeytrainerlist['Stakes Won'].str.strip(
        '$')
    if jockey > 0:
        jockeytrainerlist['Jockey'] = jockeytrainerlist['Jockey'].str.strip()
    else:
        jockeytrainerlist['Trainer'] = jockeytrainerlist['Trainer'].str.strip()
    logging.info('Jockey Trainer List: \n %s', jockeytrainerlist)
except Exception as ex:
    logging.exception(ex)
finally:
    logging.info('Close Selenium Browser')
    webCrawling.close()

# ================ Save as csv ==================
headers = ','.join(map(str, jockeytrainerlist.columns.values))

if jockey > 0:
    np.savetxt('./Raw Data/jockey1819.csv', jockeytrainerlist.round(0),
               delimiter=',', fmt='%s', header=headers, comments='')
else:
    np.savetxt('./Raw Data/trainer1819.csv', jockeytrainerlist.round(0),
               delimiter=',', fmt='%s', header=headers, comments='')

logging.info('Finished write csv')
