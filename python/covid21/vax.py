#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:07:20 2021

@author: jsibert
"""
#from socrata.authorization import Authorization
#from socrata import Socrata
#import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from covid21 import config as cv
from covid21 import Geography as GG
from covid21 import GraphicsUtilities as GU
from sodapy import Socrata
from datetime import datetime

import sys



def get_cdc_dat(update=False):
    vax_file = cv.CDC_home + 'us-vax.csv'
    print('raw_data file = ', vax_file)

    if update:
        '''
        api_end_point = 'https://data.cdc.gov/resource/8xkx-amqh.json'
        query = api_end_point+'?$limit=10000'+',$select=date'#, recip_county'
        print('query =', query)
        
        raw_data = pd.read_json(query, dtype=True)
        print(raw_data)
        print(raw_data.columns)
        '''
        
        
        '''
        raw_data = raw_data.drop(columns=['mmwr_week', 'completeness_pct', #'administered_dose1_recip',
            'administered_dose1_pop_pct', 'administered_dose1_recip_5plus',
            'administered_dose1_recip_5pluspop_pct',
            'administered_dose1_recip_12plus',
            'administered_dose1_recip_12pluspop_pct',
            'administered_dose1_recip_18plus',
            'administered_dose1_recip_18pluspop_pct',
            'administered_dose1_recip_65plus',
            'administered_dose1_recip_65pluspop_pct', #'series_complete_yes',
            'series_complete_pop_pct', 'series_complete_5plus',
            'series_complete_5pluspop_pct', 'series_complete_5to17',
            'series_complete_5to17pop_pct', 'series_complete_12plus',
            'series_complete_12pluspop_pct', 'series_complete_18plus',
            'series_complete_18pluspop_pct', 'series_complete_65plus',
            'series_complete_65pluspop_pct', 'booster_doses',
            'booster_doses_vax_pct', 'booster_doses_12plus',
            'booster_doses_12plus_vax_pct', 'booster_doses_18plus',
            'booster_doses_18plus_vax_pct', 'booster_doses_50plus',
            'booster_doses_50plus_vax_pct', 'booster_doses_65plus',
            'booster_doses_65plus_vax_pct', 'svi_ctgy',
            'series_complete_pop_pct_svi', 'series_complete_5pluspop_pct_svi',
            'series_complete_5to17pop_pct_svi', 'series_complete_12pluspop_pct_svi',
            'series_complete_18pluspop_pct_svi',
            'series_complete_65pluspop_pct_svi', 'metro_status',
            'series_complete_pop_pct_ur_equity',
            'series_complete_5pluspop_pct_ur_equity',
            'series_complete_5to17pop_pct_ur_equity',
            'series_complete_12pluspop_pct_ur_equity',
            'series_complete_18pluspop_pct_ur_equity',
            'series_complete_65pluspop_pct_ur_equity', 'booster_doses_vax_pct_svi',
            'booster_doses_12plusvax_pct_svi', 'booster_doses_18plusvax_pct_svi',
            'booster_doses_65plusvax_pct_svi', 'booster_doses_vax_pct_ur_equity',
            'booster_doses_12plusvax_pct_ur_equity',
            'booster_doses_18plusvax_pct_ur_equity',
            'booster_doses_65plusvax_pct_ur_equity', 'census2019',
            'census2019_5pluspop', 'census2019_5to17pop', 'census2019_12pluspop',
            'census2019_18pluspop', 'census2019_65pluspop'])
        print(raw_data.columns)

        raw_data = raw_data.fillna(0.0)#, inplace=True)

        '''
        APP_TOKEN = 'UwVHDRzhzc7MrtatC9AsYDcz7'
        cols='date,recip_county,recip_state,fips,administered_dose1_recip,series_complete_yes'
        client = Socrata('data.cdc.gov', APP_TOKEN)

        resource = "8xkx-amqh"
        print('getting resource:',resource)
        raw_data  = client.get(resource, limit=2000000, select=cols)
        pd.json_normalize(raw_data).to_csv(vax_file, header=True, index=False)
        print('saved raw CDC data to', vax_file)
    #    if (1): sys.exit(1)

    raw_data = pd.read_csv(vax_file, header=0)
    print(raw_data)
    raw_data = raw_data.fillna(0.0)#, inplace=True)

    raw_data['fips'].replace('UNK', np.nan, inplace=True)
    raw_data['fips'] = raw_data['fips'].fillna(0).astype(np.int64)

    udates = pd.Series(raw_data['date'].unique())
    print(udates)
    vax_len = len(raw_data)+len(udates)

    vax = pd.DataFrame(0, index=np.arange(0, vax_len),
                       columns=['date', 'mdate', 'county', 'code', 'fips', 'first', 'full'])

#   vax.astype({'date':'str', 'county':'str', 'code': 'str', 'mdate': 'float64',
#               'fips': 'int64',  'first': 'int64', 'full': 'int64'}).dtypes

#    vax['date'] = raw_data['date']
    print('reformating date field in raw_data')
  
    for k in range(len(raw_data)):
        d = raw_data['date'][k]
#        print(k,d)
#        dd = datetime.date(datetime.strptime(raw_data['date'],'%Y-%m-%dT00:00:00.000'))
        try:
            raw_data.loc[k,'date'] =  datetime.date(datetime.strptime(d,'%Y-%m-%dT00:00:00.000'))
        except:
            print('datetime error at', k,d)
    
    print('    complete')        
    '''
    try:
        raw_data['date'].apply(lambda: d, datetime.date(datetime.strptime(d,'%Y-%m-%dT00:00:00.000')))
    except Exception as ex:
        print('datetime issue:')
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
    '''    
    
    print('assembling vax data')    
    vax['date'] = raw_data['date']        
    vax['county'] = raw_data['recip_county']
    vax['code'] = raw_data['recip_state']
    vax['fips'] = raw_data['fips']
    vax['first'] = raw_data['administered_dose1_recip']
    vax['full'] = raw_data['series_complete_yes']
#   print(vax)
    vax.dropna(inplace=True)
#   print('after drop:')
    print(vax)

    NYC_fips = [36005, 36047, 36061, 36081, 36085]

#    k = len(raw_data)-1
#    for d in udates:
    for k, d0 in enumerate(udates):
    #    k += 1
        d = datetime.date(datetime.strptime(d0,'%Y-%m-%dT00:00:00.000'))
#       print('k=',k,d0,d)
        vax.loc[k, 'date'] = d
       
        vax.loc[k, 'fips'] = 36999
        vax.loc[k, 'county'] = 'New York City'
        vax.loc[k, 'code'] = 'NY'
        vax.loc[k, 'first'] = 0.0
        vax.loc[k, 'full'] = 0.0
        udmask = raw_data['date'] == d
        fipsmask = raw_data['fips'].isin(NYC_fips)
        gmask = udmask & fipsmask

        vax.loc[k, 'first'] = raw_data.loc[gmask, 'administered_dose1_recip'].sum()
        vax.loc[k, 'full'] = raw_data.loc[gmask, 'series_complete_yes'].sum()

    #   for f in NYC_fips:

#   for d in udates:
    print('    udate loop complete')        

    vax['mdate'] = pd.Series(mdates.date2num(vax['date']))
    vax['fips'] = vax['fips'].fillna(0).astype(np.int64)

#   vax = vax.astype({'date':'str',  'county':'str',  'code': 'str',  'mdate': 'float64',
#               'fips': 'int64',   'first': 'int64',  'full': 'int64'}).dtypes
#   print(vax)
    print('sorting by date and fips')
    vax = vax.sort_values(by=['date','fips'], ascending=True)

    vax_name = cv.CDC_vax
    vax.to_csv(vax_name, header=True, index=False)
    print('Augmented and sorted CDC vax data written to', vax_name)


def plot_vax(name='New York City', enclosed_by='New York', code='NY'):

    vax_name = cv.CDC_home + 'vax.csv'
    vax = pd.read_csv(vax_name, header=0, comment='#')
    print('Vax data read from', vax_name)

    print(vax.tail())

    tgeog = GG.Geography(name=name, enclosed_by=enclosed_by, code=code)
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

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.0))
    #firstDate = datetime.strptime(vax['date'][0], '%Y-%m-%d')
    print(vax['date'][0]) #, firstDate)
    GU.make_date_axis(ax)#, cv.FirstNYTDate)#, firstDate)
    ax.plot(mdate, pv)
    ax.set_ylabel('Vaccinations (%)')

    title = tgeog.name + ',  ' + tgeog.enclosed_by
    fig.text(0.5, 0.9, title, ha='center', va='bottom')

    fig.show()

#read_NYC_data()
#plot_NYC_data()
#get_cdc_dat()#True)
#plot_vax(name='Alameda',  enclosed_by='California',  code='CA')
