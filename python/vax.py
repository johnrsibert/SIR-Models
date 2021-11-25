#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:07:20 2021

@author: jsibert
"""
#from socrata.authorization import Authorization
#from socrata import Socrata
#import os

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
import sys
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
    
    fig, ax = plt.subplots(3,1,figsize=(6.5,9.0))
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
    
    tgeog = GG.Geography(name='Los Angeles',enclosed_by='California',code='CA')
    tgeog.read_nyt_data('county')
    #tgeog.print_metadata()
    #tgeog.print_data()
    #tgeog.plot_prevalence(save=True, signature=True,show_superspreader=False,per_capita=True,show_order_date = True)
    #tgeog.plot_prevalence(save=False, signature=True,show_superspreader=False,
    #        per_capita=True,show_order_date = True,cumulative = True)
    #print(tgeog.date)
    #print(tgeog.cases)
    gdf = tgeog.to_DataFrame()
    print(gdf)
    
    ax[2].plot(gdf.pdate,gdf.cases,linewidth=2)
    GU.mark_ends(ax[2],gdf.pdate,gdf.cases,'C','r')
    ax[2].set_ylabel('Cases')
    ax[2].set_ylim(0.0,1.1*gdf.cases.max())
    
    ax2 = ax[2].twinx()
    ax2.plot(gdf.pdate,gdf.deaths,linewidth=2,c='r')
    GU.mark_ends(ax2,gdf.pdate,gdf.deaths,'D','r')
    ax2.set_ylabel('Deaths')
    #ax[1].set.ylim(0,1.5e6)
    ax2.set_ylim(0.0,0.02*ax[2].get_ylim()[1])
    #ax2_lim = ax[2].get_ylim()
    #print(ax2_lim)
    #ax2.set_ylim(ax2_lim[0], 0.02*ax2_lim[1])
    fig.show()
    
def get_cdc_dat(update=False):
    '''
    vax_file = '~/Downloads/COVID-19_Vaccinations_in_the_United_States_County.csv'
    print('reading',vax_file)
    tvax = pd.read_csv(vax_file,header=0)
    print(tvax)
    print(tvax.columns)
    #[1119194 rows x 32 columns
    '''
    vax_file = cv.CDC_home + 'us-vax.csv'
   
    if update:
        api_end_point = 'https://data.cdc.gov/resource/8xkx-amqh.json'   
        query = api_end_point+'?$limit=2000000'
        raw_data = pd.read_json(query,dtype=True)
        print(raw_data)
    #    print(raw_data.columns)
    #    print(raw_data.dtypes)
    
        raw_data.to_csv(vax_file,header=True,index=False)
        print('saved CDC data to',vax_file)
   
    raw_data = pd.read_csv(vax_file,header=0)
    raw_data['fips'].replace('UNK',np.nan,inplace = True)# = tvax['FIPS'].fillna(0).astype(np.int64)
    raw_data['fips'] = raw_data['fips'].fillna(0).astype(np.int64)
    
    print(raw_data)
    print(raw_data.columns)
    
    udates = pd.Series(raw_data['date'].unique())
    print('udates:')
    print(udates)
    vax_len = len(raw_data)+len(udates)
    print(vax_len,len(raw_data),len(udates))
    
    vax=pd.DataFrame(index=np.arange(0,vax_len),
                     columns = ['date','county','code','fips','first','full'])
    vax['date'] = raw_data['date']
    vax['county'] = raw_data['recip_county']
    vax['code'] = raw_data['recip_state']
    vax['fips'] = raw_data['fips']
    vax['first'] = raw_data['administered_dose1_recip']
    vax['full'] = raw_data['series_complete_yes']

    print(vax)
    
    NYC_fips = [36005,36047,36061,36081,36085]
    
    k = len(raw_data)-1
    #for u,d in enumerate(udates):
    for d in udates:
        k += 1
    #    print(k,d)
        vax.loc[k,'date'] = d
        vax.loc[k,'fips'] = 36999
        vax.loc[k,'county'] = 'New York City'
        vax.loc[k,'code'] = 'NY'
        vax.loc[k,'first'] = 0.0
        vax.loc[k,'full'] = 0.0
        
        udmask = raw_data['date'] == d
        for f in NYC_fips:
            fipsmask = raw_data['fips'] == f
            gmask = udmask & fipsmask
                   
            if gmask.value_counts()[1] <= 1:
                vax.loc[k,'first'] += float(raw_data.loc[gmask,'administered_dose1_recip'])
                vax.loc[k,'full'] += float(raw_data.loc[gmask,'series_complete_yes'])
            #    print(raw_data.loc[gmask,['fips']])
            else:
                print('duplicate (date,fips) combination at',(d,f))
                print('   ',True,'values in gmask =',gmask.value_counts()[1])
                dupes = raw_data[gmask]
                print('    duplicates:',dupes.index)
                d0 = dupes.index[0]
      
                try:
                    vax.loc[k,'first'] += float(dupes.loc[d0,'administered_dose1_recip'])
                    vax.loc[k,'full']  += float(raw_data.loc[d0,'series_complete_yes'])
                except Exception as exception:
                    print('vax assignment filed for',(d,f,d0))
                    print('    Exception:',exception.__class__.__name__)
                    print(dupes.loc[d0])

    
    print(vax)
    
    print('sorting by date')
    vax = vax.sort_values(by='date',ascending=True)
    print(vax)
    
    
    vax_name =  cv.CDC_home + 'vax.csv'
    vax.to_csv(vax_name,header=True,index=False)
    print('Augmented vax data written to',vax_name)
    
    '''
    NYC_data = pd.DataFrame(columns=['date','fips','county','code','first','full'])
    NYC_data['date'] = udates
    #NYC_data['fips'] = 36999
    NYC_data['county'] = 'New York City'
    NYC_data['code'] = 'NY'
    NYC_data['fips'] = 36999
    NYC_data['first'] = 0.0
    NYC_data['full'] = 0.0
    print(NYC_data)
   
    NYC_fips = [36005,36047,36061,36081,36085]
    
    for k,d in enumerate(udates):
    #    print(k,d)
        udmask = raw_data['date'] == d
        for f in NYC_fips:
            fipsmask = raw_data['fips'] == f
            gmask = udmask & fipsmask
                   
            if gmask.value_counts()[1] <= 1:
                NYC_data.loc[k,'first'] += float(raw_data.loc[gmask,'administered_dose1_recip'])
                NYC_data.loc[k,'full'] += float(raw_data.loc[gmask,'series_complete_yes'])
            #    print(raw_data.loc[gmask,['fips']])
            else:
                print('duplicate (date,fips) combination at',(d,f))
                print('   ',True,'values in gmask =',gmask.value_counts()[1])
                dupes = raw_data[gmask]
                print('    duplicates:',dupes.index)
                d0 = dupes.index[0]
      
                try:
                    NYC_data.loc[k,'first'] += float(dupes.loc[d0,'administered_dose1_recip'])
                    NYC_data.loc[k,'full']  += float(raw_data.loc[d0,'series_complete_yes'])
                except Exception as exception:
                    print('NYC_data assignment filed for',(d,f,d0))
                    print('    Exception:',exception.__class__.__name__)
                    print(dupes.loc[d0])
                        
    print(NYC_data)
     
    vax=pd.DataFrame()
    vax['date'] = raw_data['date']
    vax['county'] = raw_data['recip_county']
    vax['code'] = raw_data['recip_state']
    vax['fips'] = raw_data['fips']
    vax['first'] = raw_data['administered_dose1_recip']
    vax['full'] = raw_data['series_complete_yes']

    print(vax)
    '''
     
  
    
    
    
    
#read_NYC_data()
#plot_NYC_data()
get_cdc_dat()#True)
