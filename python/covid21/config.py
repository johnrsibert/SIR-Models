#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
js_covid.py fake package to hide global variables and avoid singletons

Created on Tue Jul  7 15:08:58 2020

@author: jsibert
"""
from datetime import datetime, timedelta
import pandas as pd
#import numpy as np
import os
import matplotlib.pyplot as plt

NYT_home = '/home/other/nytimes-covid-19-data/'
NYT_counties = NYT_home + 'us-counties.csv'
NYT_states = NYT_home + 'us-states.csv'
NYT_us = NYT_home + 'us.csv'

CDC_home = '/home/other/CDC-data/'
CDC_vax = CDC_home + 'vax.csv'

cv_home  = '/home/jsibert/Projects/SIR-Models/'
Jon_path ='/home/jsibert/Projects/JonzPandemic/'
GeoIndexPath = cv_home+'GeogIndex.csv'
fit_path = cv_home + 'fits/'
dat_path = cv_home + 'dat/'
graphics_path = cv_home + 'Graphics/'
assets_path = cv_home + 'assets/'
TMB_path = cv_home + 'TMB/'

# mget http://www.bccdc.ca/Health-Info-Site/Documents/BCCDC_COVID19_Dashboard_Case_Details.csv
BCHA_path=cv_home+'BCCDC_COVID19_Dashboard_Case_Details.csv'

DexamethasoneDate = datetime.strptime('2020-06-22','%Y-%m-%d')
FirstNYTDate = datetime.strptime('2020-01-21','%Y-%m-%d')
CAOrderDate = datetime.strptime('2020-03-19','%Y-%m-%d')
mtime = os.path.getmtime(NYT_home+'us-counties.csv')
dtime = datetime.fromtimestamp(mtime)
EndOfTime = dtime.date()+timedelta(days=21)
'''
https://www.ajmc.com/view/a-timeline-of-covid-19-vaccine-developments-in-2021

https://apnews.com/article/joe-biden-coronavirus-pandemic-coronavirus-vaccine-6b624ae3a0ebdda0d91e867d59c8ca46?
'''
PfizerEUADate = datetime.strptime('2021-01-06','%Y-%m-%d')
HalfMillionShotDate = datetime.strptime('2021-02-25','%Y-%m-%d')
IndependenceDay = datetime.strptime('2021-07-04','%Y-%m-%d')

GeoIndex = pd.read_csv(GeoIndexPath,header=0,comment='#')
#GeoIndex['fips'] = GeoIndex['fips'].fillna(0).astype(np.int64) 
 
#print(GeoIndex.head())
nyt_county_dat = pd.read_csv(NYT_counties,header=0)
#print(nyt_county_dat.head())
nyt_state_dat = pd.read_csv(NYT_states,header=0)
#print(nyt_state_dat)
cdc_vax_dat = pd.read_csv(CDC_vax,header=0)
#print(cdc_vax_dat.tail())

eps = 1e-5
 
# "temporary" workaround issue with pyreadr.read_r(...)
# reading TMB standard error objects
pyreadr_kludge = False

plt.style.use('file:///home/jsibert/.config/matplotlib/john.mplstyle')

