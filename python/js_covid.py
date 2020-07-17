#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
js_covid.py fake package to hide global variables and avoid singletons

Created on Tue Jul  7 15:08:58 2020

@author: jsibert
"""
from datetime import datetime

# mget http://www.bccdc.ca/Health-Info-Site/Documents/BCCDC_COVID19_Dashboard_Case_Details.csv

NYT_home = '/home/other/nytimes-covid-19-data/'
NYT_counties = NYT_home + 'us-counties.csv'
NYT_states = NYT_home + 'us-states.csv'
NYT_us = NYT_home + 'us.csv'

cv_home = '/home/jsibert/Projects/SIR-Models/'
census_data_path = cv_home+'co-est2019-pop.csv'
large_county_path = cv_home + 'county-state.csv'
fit_path = cv_home + 'fits/'
dat_path = cv_home + 'dat/'
graphics_path = cv_home + 'Graphics/'
TMB_path = cv_home + 'TMB/'

FirstNYTDate = datetime.strptime('2020-01-21','%Y-%m-%d')
CAOrderDate = datetime.strptime('2020-03-19','%Y-%m-%d')
EndOfTime = datetime.strptime('2020-07-31','%Y-%m-%d')
