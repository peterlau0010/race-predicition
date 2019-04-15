from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
import bs4
import logging
import time
import re
import numpy
import pandas

class WebCrawling:
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.INFO)
    
    def __init__(self):
        self.browser = webdriver.Safari()

    def close(self):
        self.browser.quit()

    def request(self, url):
        self.browser.get(url)
        time.sleep(5)
        html_source = bs4.BeautifulSoup(self.browser.page_source,'lxml')
        return html_source
    
    def reFormatText(self, text):
        #Trim space, remove line break, remove double space
        return re.sub(' +', ' ',text.replace('\n', ' ').replace('\r', '').strip())

    def getHistory(self,day,racecource,raceno):
        url_base = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?'
        url_raceDate = 'RaceDate='+ day
        url_raceCourse = '&Racecourse=' + racecource
        url_raceNo = '&RaceNo=' + raceno
        url = url_base + url_raceDate + url_raceCourse + url_raceNo
        
        logging.info('Access Url: %s', url)

        html_source = self.request(url)

        table = html_source.find(name='table',class_='f_tac table_bd draggable')
        history= []

        for t in table.find_all(name = 'td'):
            history.append(webCrawling.reFormatText(t.text))

        logging.debug('Before reFormat, Shape of Histroy: %s', numpy.shape(history))
        history = numpy.reshape(history,(-1,12))
        logging.debug('After reFormat, Shape of Histroy: %s', numpy.shape(history))
        logging.debug('Remove Header')
        history = numpy.delete(history,(0),axis=0)

        return history

history = numpy.zeros(shape=(0,12))
matchCSV = pandas.read_csv('matchDate.csv', engine='python',sep = ',')
logging.info(matchCSV)

# initial webCrawling
# webCrawling = WebCrawling()

# for i in range(1,9):
#     temp_history = webCrawling.getHistory('2019/04/10','HV',str(i))
#     history = numpy.concatenate((history,temp_history),axis=0)




# webCrawling.close()

# numpy.savetxt('history.csv', history, delimiter=',',fmt='%s')