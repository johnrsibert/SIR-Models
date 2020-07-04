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
            
    def get_pdate(self):
        if (self.pdate):
            return(pdate)

        else:
            self.pdate = []
            for d in self.date:
                self.pdate.append(mdates.date2num(datetime.strptime(d,'%Y-%m-%d').date()))
            return(self.pdate)

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

   #    tmp = dat[County_rows]['date'].map(Strptime)
   #    self.pdate  = np.array(mdates.date2num(tmp))
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

    def dow_count(self,mult = 1000.0):
        """
        Accumulate first differences of cases and deaths by day of week
        Scale by mult/population
        """
        names = ['Mo','Tu','We','Th','Fr','Sa','Su','moniker']
        Ccount = pd.Series(0.0,index=names)
        Dcount = pd.Series(0.0,index=names)
        d1_cases = np.diff(self.cases)
        d1_deaths = np.diff(self.deaths)
    
        for k in range(0,len(self.date)-1):            
            j = datetime.strptime(self.date[k],'%Y-%m-%d').weekday()
            Ccount[j] += d1_cases[k]
            Dcount[j] += d1_deaths[k]

        scale = mult/self.population
        Ccount = Ccount * scale
        Dcount = Dcount * scale
        Ccount['moniker'] = self.moniker
        Dcount['moniker'] = self.moniker
        counts = [Ccount,Dcount]
        return(counts)
        
def make_dow_table(mult=1000):        
    """
    accumulate dow counts by geography as table rows
    """
    names = ['Mo','Tu','We','Th','Fr','Sa','Su']
    csdat = pd.read_csv(large_county_path,header=0)
    cases_count  = pd.DataFrame(dtype=float)
    deaths_count = pd.DataFrame(dtype=float)
   
    for cs in range(0,len(csdat)):
        tmpG = Geography(csdat['county'][cs],csdat['state'][cs],csdat['ST'][cs])
        tmpG.read_nyt_data('county')
        row = tmpG.dow_count(mult)
    #   print(tmpG.moniker,type(row),len(row))
        cases_count  = cases_count.append(row[0],ignore_index=True)
        deaths_count = deaths_count.append(row[1],ignore_index=True)
      
    cases_count = cases_count.set_index('moniker')
    deaths_count = deaths_count.set_index('moniker')
    cases_count = cases_count.reindex(columns=names)
    deaths_count = deaths_count.reindex(columns=names)
    return[cases_count,deaths_count]
    
def plot_dow_boxes(mult=1000):
    counts = make_dow_table(mult)
    labels = counts[0].columns
    title = str(counts[0].shape[0])+' most populous US counties'

    fig, ax = plt.subplots(2,1,figsize=(6.5,4.5))
    fig.text(0.5,0.9,title,ha='center',va='bottom')
    ax[0].set_ylabel('Cases'+' per '+str(mult))
    ax[0].boxplot(counts[0].transpose(),labels=labels)
    ax[1].set_ylabel('Deaths'+' per '+str(mult))
    ax[1].boxplot(counts[1].transpose(),labels=labels)
  
    plt.savefig('days_of_week.png',dpi=300)
    plt.show()
  
# --------------------------------------------------       
alam = Geography('Alameda','California','CA')
alam.read_nyt_data('county')
alam.get_pdate()
alam.print_data()
#hono = Geography('Honolulu','Hawaii','HI')
#hono.read_nyt_data('county')
#test.print_metadata()
#test.print_data()
#plot_dow_boxes()
