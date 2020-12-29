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
import matplotlib.dates as mdates
# mget http://www.bccdc.ca/Health-Info-Site/Documents/BCCDC_COVID19_Dashboard_Case_Details.csv

NYT_home = '/home/other/nytimes-covid-19-data/'
NYT_counties = NYT_home + 'us-counties.csv'
NYT_states = NYT_home + 'us-states.csv'
NYT_us = NYT_home + 'us.csv'

cv_home = '/home/jsibert/Projects/SIR-Models/'
#census_data_path = cv_home+'co-est2019-pop.csv'
census_data_path = cv_home+'nyt_census.csv'
large_county_path = cv_home + 'county-state.csv'
fit_path = cv_home + 'fits/'
dat_path = cv_home + 'dat/'
graphics_path = cv_home + 'Graphics/'
assets_path = cv_home + 'assets/'
TMB_path = cv_home + 'TMB/'

BCHA_path=cv_home+'BCCDC_COVID19_Dashboard_Case_Details.csv'

FirstNYTDate = datetime.strptime('2020-01-21','%Y-%m-%d')
CAOrderDate = datetime.strptime('2020-03-19','%Y-%m-%d')
#EndOfTime = datetime.strptime('2020-10-28','%Y-%m-%d')
mtime = os.path.getmtime(NYT_home+'us-counties.csv')
dtime = datetime.fromtimestamp(mtime)
EndOfTime = dtime.date()+timedelta(days=14)
#print('EndOfTime:',EndOfTime,mdates.date2num(EndOfTime))
   
population_dat = pd.DataFrame(None)
nyt_county_dat = pd.DataFrame(None)

# "temporary" workaround issue with pyreadr.read_r(...)
# reading TMB standard error objects
pyreadr_kludge = False

import pandas as pd
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc
import numpy as np
import os
import sys
import pyreadr
from io import StringIO 
from io import BytesIO
import base64
import scipy.stats as stats
from sigfig import round
from tabulate import tabulate
from collections import OrderedDict
import glob
import re
import statistics

plt.style.use('file:///home/jsibert/.config/matplotlib/john.mplstyle')

