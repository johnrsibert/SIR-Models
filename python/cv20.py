#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taking a more OO approach to the code in cv19.py

Created on Thu Jul  2 09:04:11 2020

@author: jsibert
"""

import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc
import numpy as np
import os
import pyreadr
from io import StringIO
import scipy.stats as stats
from sigfig import round
from tabulate import tabulate
from collections import OrderedDict

NYT_home = '/home/other/nytimes-covid-19-data/'
NYT_counties = NYT_home + 'us-counties.csv'
NYT_states = NYT_home + 'us-states.csv'
NYT_us = NYT_home + 'us.csv'
cv_home = '/home/jsibert/Projects/SIR-Models/'
census_data_path = cv_home+'co-est2019-pop.csv'
fit_path = cv_home + 'fits/'
dat_path = cv_home + 'tests/'

def Strptime(x):
    """
    wrapper for datetime.strptime callable by map(..)
    """
    y = datetime.strptime(x,'%Y-%m-%d')
    return(y)

class Geography:

    def __init__(self,name,enclosed_by,code):
        self.gtype = None
        self.name = name 
        self.enclosed_by = enclosed_by
        self.code = code
        self.population = None
        self.source = None
        self.moniker = name+code
        self.TMB_fit = None
        self.ADMB_fit = None
        self.dat_file = dat_path+self.moniker+'.dat'
        self.updated = None
        self.date0 = None
        self.Date = None
        self.cases = None
        self.deaths = None
        
    def print(self):
        print(self)
        
        
    def print_metadata(self):
        print(self.gtype)
        print(self.name)
        print(self.enclosed_by)
        print(self.code)
        print(self.population)
        print(self.source)
        print(self.moniker)
        print(self.TMB_fit)
        print(self.ADMB_fit)
        print(self.dat_file)
        print(self.updated)

    def print_data(self):
        print('Date','cases','deaths')
        for k in range(0,len(self.Date)):
            print(self.Date[k],self.cases[k],self.deaths[k])
            
            
    def get_county_pop(self):
        """
        Reads US Census populations estimates for 2019
        Subset of orginal csv file saved in UTF-8 format. 
        The word 'County' stipped from county names.

        nyc_pop is the sum of census estimates for New York, Kings, Queens
        Bronx, and Richmond counties to be compatible with NYTimes data
        """
        
        nyc_pop = 26161672
        if (self.name == 'New York City'):
            return(nyc_pop)

        dat = pd.read_csv(census_data_path,header=0)
        state_filter = dat['state'].isin([self.enclosed_by])
        county_filter = dat['county'].isin([self.name])
        County_rows = state_filter & county_filter
        population = int(pd.to_numeric(dat[County_rows]['population'].values))
        return(population)   
        
    def read_nyt_data(self,gtype):
        self.gtype = gtype
        if (gtype == 'county'):
            cspath = NYT_counties
        else:
            sys.exit('No data for',gtype)
        
        
        dat = pd.read_csv(cspath,header=0)

        mtime = os.path.getmtime(cspath)
        dtime = datetime.fromtimestamp(mtime)
        self.updated = str(dtime.date())
        self.population = self.get_county_pop()
   #     dat['county'][self.name],dat['state'][self.enclosed_by])
    
        state_filter = dat['state'].isin([self.enclosed_by])
        county_filter = dat['county'].isin([self.name])
        County_rows = state_filter & county_filter
        if (len(County_rows) < 1):
            sys.exit(' * * * no recores fround for',self.name,self.surrounded_by)

        tmp = dat[County_rows]['date'].map(Strptime)
        tmp = np.array(tmp)
#        print(tmp)
#        print('d0: ',str(tmp[0]),datetime.strptime(str(tmp[0]),'%Y-%m-%d'))
    
        r0 = dat['date'][County_rows].index[0]
        self.date0 = dat['date'][r0]

        self.Date = np.array(mdates.date2num(tmp))
        self.cases = np.array(dat[County_rows]['cases'])
        self.deaths = np.array(dat[County_rows]['deaths'])
        
    def write_dat_file(self):
        print(len(self.Date),'records found for',self.name,self.enclosed_by)
        O = open(self.dat_file,'w')
        O.write('# county\n %s\n'%(self.moniker))
        O.write('# updateed from https://github.com/nytimes/covid-19-data.git\n')
        O.write(' %s\n'%self.updated)
        O.write('# population (N0)\n %10d\n'%self.population)
        O.write('# date zero\n %s\n'%self.date0)
    #   print(dat[County_rows])
        ntime = len(self.Date)-1
    #   print('----------------ntime',ntime)
        O.write('# ntime\n %4d\n'%ntime)
    #   print(dat[County_rows].index)
    #   print('date','index','cases','deaths')
        O.write('#%6s %5s\n'%('cases','deaths'))
        for r in range(0,len(self.Date)):
            O.write('%7d %6d\n'%(self.cases[r],self.deaths[r]))

        
test = Geography('Alameda','California','CA')
test.read_nyt_data('county')
test.print_metadata()
#test.print_data()

