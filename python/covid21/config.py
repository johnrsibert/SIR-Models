#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
js_covid.py fake package to hide global variables and avoid singletons

Created on Tue Jul  7 15:08:58 2020

@author: jsibert
"""
from datetime import datetime, timedelta
import pandas as pd
import os
import matplotlib.pyplot as plt

NYT_home = '/home/other/nytimes-covid-19-data/'
NYT_counties = NYT_home + 'us-counties.csv'
NYT_states = NYT_home + 'us-states.csv'
NYT_us = NYT_home + 'us.csv'

cv_home = '/home/jsibert/Projects/SIR-Models/'
GeoIndexPath = cv_home+'GeogIndex.csv'
fit_path = cv_home + 'fits/'
dat_path = cv_home + 'dat/'
graphics_path = cv_home + 'Graphics/'
assets_path = cv_home + 'assets/'
TMB_path = cv_home + 'TMB/'

# mget http://www.bccdc.ca/Health-Info-Site/Documents/BCCDC_COVID19_Dashboard_Case_Details.csv
BCHA_path=cv_home+'BCCDC_COVID19_Dashboard_Case_Details.csv'

FirstNYTDate = datetime.strptime('2020-01-21','%Y-%m-%d')
CAOrderDate = datetime.strptime('2020-03-19','%Y-%m-%d')
mtime = os.path.getmtime(NYT_home+'us-counties.csv')
dtime = datetime.fromtimestamp(mtime)
EndOfTime = dtime.date()+timedelta(days=14)
   
GeoIndex = pd.read_csv(GeoIndexPath,header=0,comment='#')
#print(GeoIndex.head())
nyt_county_dat = pd.read_csv(NYT_counties,header=0)
#print(nyt_county_dat.head())

eps = 1e-5
 
# "temporary" workaround issue with pyreadr.read_r(...)
# reading TMB standard error objects
pyreadr_kludge = False

plt.style.use('file:///home/jsibert/.config/matplotlib/john.mplstyle')

