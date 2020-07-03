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
large_county_path = cv_home + 'county-state.csv'
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
        self.date = None
        self.cases = None
        self.deaths = None
        self.pdate = None # date for plotting on x-axis
        
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
        print(self.date0)

    def print_data(self):
        print('date','cases','deaths','pdate')
        for k in range(0,len(self.date)):
            print(self.date[k],self.cases[k],self.deaths[k],self.pdate[k])
            
            
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
    
        state_filter = dat['state'].isin([self.enclosed_by])
        county_filter = dat['county'].isin([self.name])
        County_rows = state_filter & county_filter
        if (len(County_rows) < 1):
            sys.exit(' * * * no recores fround for',self.name,self.surrounded_by)

        tmp = dat[County_rows]['date'].map(Strptime)
        self.pdate  = np.array(mdates.date2num(tmp))
        self.cases  = np.array(dat[County_rows]['cases'])
        self.deaths = np.array(dat[County_rows]['deaths'])
        self.date   = np.array(dat[County_rows]['date'])
        self.date0 = self.date[0]
        
    def write_dat_file(self):
        print(len(self.Date),'records found for',self.name,self.enclosed_by)
        O = open(self.dat_file,'w')
        O.write('# county\n %s\n'%(self.moniker))
        O.write('# updateed from https://github.com/nytimes/covid-19-data.git\n')
        O.write(' %s\n'%self.updated)
        O.write('# population (N0)\n %10d\n'%self.population)
        O.write('# date zero\n %s\n'%self.date0)
        ntime = len(self.Date)-1
        O.write('# ntime\n %4d\n'%ntime)
        O.write('#%6s %5s\n'%('cases','deaths'))
        for r in range(0,len(self.Date)):
            O.write('%7d %6d\n'%(self.cases[r],self.deaths[r]))

    def dow_count(self):
        names = ['Mo','Tu','We','Th','Fr','Sa','Su','moniker']
        count = pd.Series(0.0,index=names)
    #    count = np.array(7*[0.0])
    #    print(count)
    
        for k in range(0,len(self.date)):            
            j = datetime.strptime(self.date[k],'%Y-%m-%d').weekday()
            count[j] += self.cases[k]

        count['moniker'] = self.moniker
     #   print(count,sum(count),sum(self.cases))
     #   for j in range(0,len(count)):
     #       print(j,names[j],count[j])
        return(count) 
        
def make_dow_table():        
    """
    """
    names = ['Mo','Tu','We','Th','Fr','Sa','Su']
    total_count = pd.DataFrame(dtype=float)
    csdat = pd.read_csv(large_county_path,header=0)
    
    for cs in range(0,10): #len(csdat)):
        tmpG = Geography(csdat['county'][cs],csdat['state'][cs],csdat['ST'][cs])
        tmpG.read_nyt_data('county')
        row = pd.Series(tmpG.dow_count())
    #    print(tmpG.moniker,row)
        total_count = total_count.append(row,ignore_index=True)
        
    print(total_count.shape)
    print(total_count)   
    total_count = total_count.set_index('moniker')
    print(total_count)  
    total_count = total_count.reindex(columns=names)
    print(total_count)
    print(total_count.loc['MaricopaAZ'])    
   
        
alam = Geography('Alameda','California','CA')
alam.read_nyt_data('county')
#hono = Geography('Honolulu','Hawaii','HI')
#hono.read_nyt_data('county')
#test.print_metadata()
#test.print_data()
make_dow_table()