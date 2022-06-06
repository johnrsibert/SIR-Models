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
#from datetime import date, datetime, timedelta
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
   
def get_cdc_dat(update=False):
    vax_file = cv.CDC_home + 'us-vax.csv'
    print('vax_file = ',vax_file)
   
    if update:
        #                https://data.cdc.gov/resource/8xkx-amqh.json
        api_end_point = 'https://data.cdc.gov/resource/8xkx-amqh.json'   
      #  query = api_end_point+'?$limit=2000000'
        query = api_end_point+'?$limit=2000000'
        
        print('query =',query)
        raw_data = pd.read_json(query,dtype=True)
    #    raw_data = pd.read_json('https://data.cdc.gov/resource/8xkx-amqh.json')
        
   #    print('raw_data:')
   #    print(raw_data)
        raw_data.fillna(0.0,inplace=True)
        print(raw_data.tail(n=50))
  #      print('zero:',raw_data.iloc[0])
        print('columns')
        print(raw_data.columns)
        print('dtypes')
        print(raw_data.dtypes)
    
        raw_data.to_csv(vax_file,header=True,index=False)
        print('saved CDC data to',vax_file)
   
    raw_data = pd.read_csv(vax_file,header=0)
    raw_data['fips'].replace('UNK',np.nan,inplace = True)# = tvax['FIPS'].fillna(0).astype(np.int64)
    raw_data['fips'] = raw_data['fips'].fillna(0).astype(np.int64)
    
#   print(raw_data)
#   print(raw_data.columns)
    
    udates = pd.Series(raw_data['date'].unique())
    print('udates:',len(udates))
    print(udates)
    vax_len = len(raw_data)+len(udates)
    print(vax_len,len(raw_data),len(udates))
    
    vax=pd.DataFrame(0,index=np.arange(0,vax_len),
                     columns = ['date', 'mdate', 'county', 'code', 'fips', 'first', 'full'])

#   vax.astype({'date':'str', 'county':'str', 'code': 'str', 'mdate': 'float64',
#               'fips': 'int64',  'first': 'int64', 'full': 'int64'}).dtypes
#   {'col1': 'int32'}
#   print('init vax:')
#   print(vax)
    vax['date'] = raw_data['date']
    vax['county'] = raw_data['recip_county']
    vax['code'] = raw_data['recip_state']
    vax['fips'] = raw_data['fips']
    vax['first'] = raw_data['administered_dose1_recip']
    vax['full'] = raw_data['series_complete_yes']

#   print('udate vax:')
#   print(vax)
#   if (1): sys.exit(1)
    
    NYC_fips = [36005,36047,36061,36081,36085]
    print(NYC_fips)
    
    k = len(raw_data)-1
    print('k =',k)
    for d in udates:
        k += 1
#       print('k =',k,'d =',d)
        vax.loc[k,'date'] = d
        vax.loc[k,'fips'] = 36999
        vax.loc[k,'county'] = 'New York City'
        vax.loc[k,'code'] = 'NY'
        vax.loc[k,'first'] = 0.0
        vax.loc[k,'full'] = 0.0
#       print('vax',d,':')
#       print(vax)
        udmask = raw_data['date'] == d
    #   print(raw_data['fips'])
    #   print(NYC_fips)
        fipsmask = raw_data['fips'].isin(NYC_fips)
    #   print(fipsmask)
        gmask = udmask & fipsmask
#       print('gmask:')
    #   print(gmask)
#       print(raw_data[gmask])
        
#       print(raw_data.loc[gmask,'series_complete_yes'].sum())
#       print(raw_data.loc[gmask,'census2019_65pluspop'].sum())

        vax.loc[k,'first'] = raw_data.loc[gmask,'administered_dose1_recip'].sum()
        vax.loc[k,'full'] = raw_data.loc[gmask,'series_complete_yes'].sum()
        
        '''       
        for f in NYC_fips:
            fipsmask = raw_data['fips'] == f
            print('f =',f,'k =',k,'fipsmask =',fipsmask,raw_data.loc[fipsmask])
            gmask = udmask & fipsmask
            print('gmask:')
            print('    value_counts:',len(gmask.value_counts()))
            print(gmask.value_counts())
            print(gmask.sum())
            print("raw_data.loc[gmask,'administered_dose1_recip']")
            print(raw_data.loc[gmask,'administered_dose1_recip'])
            vax.loc[k,'first'] += float(raw_data.loc[gmask,'administered_dose1_recip'])
            vax.loc[k,'full'] += float(raw_data.loc[gmask,'series_complete_yes'])
       #    if gmask.value_counts()[1] <= 1:
            if (len(gmask.value_counts())) >= 2:
                print(raw_data.loc[gmask,['fips']])
                print('     k =',k)
                print('    ', raw_data.loc[gmask,'administered_dose1_recip'])                
                vax.loc[k,'first'] += float(raw_data.loc[gmask,'administered_dose1_recip'])
                vax.loc[k,'full'] += float(raw_data.loc[gmask,'series_complete_yes'])
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
        '''       
    #   for f in NYC_fips:
        print('end of NYC_fips loop')

#   for d in udates:
    print('end of udates loop')

#   print('vax 0:')
#   print(vax)
    vax['mdate'] = pd.Series(mdates.date2num(vax['date']))
    vax['fips'] = vax['fips'].fillna(0).astype(np.int64)
#   print('vax 1:')
#   print(vax)
    
#   vax = vax.astype({'date':'str', 'county':'str', 'code': 'str', 'mdate': 'float64',
#               'fips': 'int64',  'first': 'int64', 'full': 'int64'}).dtypes

#   {'col1': 'int32'}
    print('sorting by date')
    vax = vax.sort_values(by='date',ascending=True)
#   print(vax)
    
    
    vax_name =  cv.CDC_home + 'vax.csv'
    vax.to_csv(vax_name,header=True,index=False)
    print('Augmented vax data written to',vax_name)
    
     
def plot_vax(name='New York City',enclosed_by='New York',code='NY'):

    vax_name =  cv.CDC_home + 'vax.csv'
    vax = pd.read_csv(vax_name,header=0,comment='#')
    print('Vax data read from',vax_name)
        
    print(vax.tail())
          
    tgeog = GG.Geography(name=name,enclosed_by=enclosed_by,code=code)
    tgeog.read_nyt_data('county')
    tgeog.print_metadata()
  
    pop = tgeog.population
    fips = tgeog.fips
    mult = 100.0  
    print(tgeog.moniker, fips, pop, mult)
    
    fmask = vax['fips'] == fips   
    mdate = vax['mdate'][fmask]
    print(mdate)
    print(vax[fmask])
    pv = mult*pd.Series(vax['first'][fmask])/pop
    print(pv.tail())
         
    fig, ax = plt.subplots(1,1,figsize=(6.5,3.0))
    #firstDate = datetime.strptime(vax['date'][0],'%Y-%m-%d')
    print(vax['date'][0]) #,firstDate)
    GU.make_date_axis(ax)#,cv.FirstNYTDate)#,firstDate)
    ax.plot(mdate,pv)
    ax.set_ylabel('Vaccinations (%)')
    
    title = tgeog.name + ', ' + tgeog.enclosed_by
    fig.text(0.5,0.9,title,ha='center',va='bottom')

    fig.show()    
    
#read_NYC_data()
#plot_NYC_data()
#get_cdc_dat()#True)
#plot_vax(name='Alameda',enclosed_by='California',code='CA')
