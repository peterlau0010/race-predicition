from selenium import webdriver
import bs4
import logging

class WebCrawling:
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)


    def request(self, url):
        driver = webdriver.Safari()
        driver.get(url)
        # driver.navigate.to("http://www.python.org")
        

    def extractText(self, fullText):
        return 

url = 'https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2019/04/10&Racecourse=HV&RaceNo=2'


webCrawling = WebCrawling()
webCrawling.request(url)