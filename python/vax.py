#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:07:20 2021

@author: jsibert
"""
from covid21 import config as cv
from covid21 import Geography as GG
#from covid21 import Fit as FF
from covid21 import GraphicsUtilities as GU
#from covid21 import CFR

#rom numpy import errstate,isneginf #,array
import pandas as pd
from datetime import date, datetime, timedelta
#from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#import matplotlib
#from matplotlib import rc
import numpy as np
#import os
#import sys
#import pyreadr
#from io import StringIO 
#from io import BytesIO
#import base64
#import scipy.stats as stats
#import time

#https://data.cdc.gov/api/views/8xkx-amqh/rows.csv?accessType=DOWNLOAD

#   BC_cases_file = 'BCCDC_COVID19_Dashboard_Case_Details.csv'
#   cmd = 'wget http://www.bccdc.ca/Health-Info-Site/Documents/' + BC_cases_file +\
#        ' -O '+cv.cv_home+BC_cases_file
#   print(cmd)
#   os.system(cmd)

# wget https://data.cdc.gov/api/views/8xkx-amqh/rows.csv

def read_vax():
    vax_file = '~/Downloads/COVID-19_Vaccinations_in_the_United_States_County.csv'
    print('reading',vax_file)
    tvax = pd.read_csv(vax_file,header=0)
    print(tvax)
    
    '''
    pdate = [0.0]*len(tvax)
    c = 0
    for k,d in enumerate(tvax['Date']):
        c += 1
        pdate[k] = mdates.date2num(datetime.strptime(d,'%m/%d/%Y').date())
        if k <= 1:
            print(c,k,d,pdate[k])
    '''
    pdate = pd.to_datetime(tvax['Date'],format ='%m/%d/%Y' )
#    dat['fips'] = dat['fips'].fillna(0).astype(np.int64)
    tvax['FIPS'].replace('UNK',np.nan,inplace = True)# = tvax['FIPS'].fillna(0).astype(np.int64)
    tvax['FIPS'] = tvax['FIPS'].fillna(0).astype(np.int64)

    #tvax['Date'] = pdate
    #print(tvax)
    
   
    print(tvax.columns)

    vax=pd.DataFrame()#columns=['date','firstjab','full'])
    '''
     elf.pdate = []
            for d in self.date:
                self.pdate.append(mdates.date2num(datetime.strptime(d,'%Y-%m-%d').date()))
            return(self.pdate)
    print(vax)
    pdate = [0.0]*len(tvax)
    for k,d in enumerate(tvax['Date']):
    #    print(k,d)
        pdate[k] = mdates.date2num(datetime.strptime(d,'%m/%d/%Y').date())
    '''  
    
    vax['date'] = pdate
#   vax['Date'] = tvax['Date']
    vax['county'] = tvax['Recip_County']
    vax['code'] = tvax['Recip_State']
    vax['fips'] = tvax['FIPS']

    vax['first'] = tvax['Administered_Dose1_Recip']
    vax['full'] = tvax['Series_Complete_Yes']
    print('sorting by date')
    vax = vax.sort_values(by='date',ascending=True)
    print(vax)
    
    vax.to_csv('vax.csv',header=True,index=False)
    print('saved vax.csv')
    
def plot_vax():
    vax = pd.read_csv('vax.csv')   
    print(vax)
    
    fig, ax = plt.subplots(2,1,figsize=(6.5,6.0))
 #   firstDate = datetime.strptime(vax['date'][0],'%Y-%m-%d')
    print(vax['date'][0]) #,firstDate)
    for a in ax:
        GU.make_date_axis(a)#,cv.FirstNYTDate)#,firstDate)
        
        
 #  Los Angeles
    pop = 10039107.0
    mult = 100.0
       
    cmask = vax['fips'] == 6037
    print(cmask)
    print(vax[cmask])
    mdate = pd.Series(mdates.date2num(vax['date'][cmask]))
    print(mdate)
    #print(vax[cmask].index[0])#vax[cmask].index)
    print('mdate',len(mdate),mdate.min(),mdate.max())
      
    ax[0].plot(mdate,mult*vax['first'][cmask]/pop,linewidth=2)
    GU.mark_ends(ax[0],mdate, mult*vax['first'][cmask]/pop, 'One','r')
    ax[0].plot(mdate,mult*vax['full'][cmask]/pop,linewidth=2)
    GU.mark_ends(ax[0],mdate,mult*vax['full'][cmask]/pop,' Full','r')
    ax[0].set_ylabel('Vaccination rate (%)')
    
    #delta = np.diff(gdf.iloc[:,a])
    dfirst = pd.Series(np.diff(vax['first'][cmask]))
    ax[1].plot(mdate[1:],dfirst,linewidth=2)
    GU.mark_ends(ax[1],mdate[1:],dfirst, 'One','r')
    
    dfull = pd.Series(np.diff(vax['full'][cmask]))
    ax[1].plot(mdate[1:],dfull,linewidth=1)
    GU.mark_ends(ax[1],mdate[1:],dfull, ' Full','r')
    ax[1].set_ylabel('Vaccinations per day')
    
    fig.show()
    
    
#read_vax()
plot_vax()