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




def get_cdc_dat(update=False):
    vax_file = cv.CDC_home + 'us-vax.csv'
    print('vax_file = ', vax_file)

    if update:
        api_end_point = 'https://data.cdc.gov/resource/8xkx-amqh.json'
        query = api_end_point+'?$limit=2000000'

        print('query =', query)
        raw_data = pd.read_json(query, dtype=True)

        raw_data.fillna(0.0, inplace=True)

        raw_data.to_csv(vax_file, header=True, index=False)
        print('saved raw CDC data to', vax_file)

    raw_data = pd.read_csv(vax_file, header=0)
    raw_data['fips'].replace('UNK', np.nan, inplace=True)
    raw_data['fips'] = raw_data['fips'].fillna(0).astype(np.int64)

    udates = pd.Series(raw_data['date'].unique())
    vax_len = len(raw_data)+len(udates)

    vax = pd.DataFrame(0, index=np.arange(0, vax_len),
                       columns=['date', 'mdate', 'county', 'code', 'fips', 'first', 'full'])

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

    NYC_fips = [36005, 36047, 36061, 36081, 36085]

    k = len(raw_data)-1
    for d in udates:
        k += 1
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

    vax['mdate'] = pd.Series(mdates.date2num(vax['date']))
    vax['fips'] = vax['fips'].fillna(0).astype(np.int64)

#   vax = vax.astype({'date':'str',  'county':'str',  'code': 'str',  'mdate': 'float64',
#               'fips': 'int64',   'first': 'int64',  'full': 'int64'}).dtypes

    print('sorting by date')
    vax = vax.sort_values(by='date', ascending=True)

    vax_name = cv.CDC_home + 'vax.csv'
    vax.to_csv(vax_name, header=True, index=False)
    print('Augmented vax data written to', vax_name)


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
