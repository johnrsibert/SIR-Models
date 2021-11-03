#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jsibert
"""


from covid21 import config as cv
from covid21 import Geography as GG
from covid21 import Fit as FF
from covid21 import GraphicsUtilities as GU
from covid21 import CFR

from numpy import errstate,isneginf #,array
import pandas as pd
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from matplotlib import rc
import numpy as np
import os
import sys
import pyreadr
from io import StringIO 
from io import BytesIO
import base64
import scipy.stats as stats
import time
from joypy import joyplot
from matplotlib import cm


#from sigfig import round
#from tabulate import tabulate
#from collections import OrderedDict
#import glob
#import re
#import statistics


def CFR_comp(nG=5, w = 14):
    dat = cv.nyt_county_dat 
    dat['fips'] = dat['fips'].fillna(0).astype(np.int64)
    NYC_mask = dat['county'] == 'New York City' 
    dat.loc[NYC_mask,'fips'] = 36999

    # creat vector of dates from first reported death untill ...
    k = 0
    while dat['deaths'][k]< 1.0:
        k += 1
#   d1 = cv.nyt_county_dat['date'].iloc[k] # first death
#   d1 = '2020-03-01' # avoid axis label on 2020-02-29
#   d2 = cv.nyt_county_dat['date'].iloc[-1] # last record
    d1 = '2020-01-01'
    d2 = '2020-04-30'
    date_list = pd.DatetimeIndex(pd.date_range(d1,d2),freq='D')
    print('processing',nG,'geographies and',len(date_list),'dates:')
    ndate = len(date_list)
    size = nG*ndate

#   create stacked date vector
    all_dates = pd.Series(['']*size)
    k2 = -1
    for g in range(0,nG):
        k1 = k2 + 1
        k2 = k1 + ndate -1
        for d in range(0,ndate):
            k = k1 + d
            all_dates[k] = datetime.strftime(date_list[d],'%Y-%m-%d')#date_list[d]

    
#   print(all_dates)
    print('Computing',size,'CFR estimates for',nG,'geographies and',len(date_list),'dates')
#   if (1): sys.exit(1)

    # extract cumulative cases and deaths from NYT data
    # for first (ie largest) nG Geographies
    cases = pd.DataFrame(0.0,columns=np.arange(0,nG), index = date_list)
    deaths = pd.DataFrame(0.0,columns=np.arange(0,nG), index = date_list)
    gg = cv.GeoIndex
    for g in range(0,nG):
        fips = gg['fips'].iloc[g]
        print(g,':',gg['county'].iloc[g] ,fips)
        fips_mask = dat['fips'] == fips
        fips_entries = fips_mask.value_counts()[1]
   
        gindex =  dat[fips_mask].index
        for k in gindex:
            tmp = dat.loc[k]
            date = tmp['date']
            if date in date_list:
                cases.loc[date,g]  = tmp['cases']
                deaths.loc[date,g] = tmp['deaths']

 #  print(deaths)
 #  print(cases)

    # compute daily cases and deaths by first differences
    dcases = pd.DataFrame(0.0,columns=np.arange(0,nG), index = date_list)
    ddeaths = pd.DataFrame(0.0,columns=np.arange(0,nG), index = date_list)
    for g in range(0,nG):
        dcases[g][1:]  = np.diff(cases[g])
        ddeaths[g][1:] = np.diff(deaths[g])

 #  print('dcases:')
 #  print(dcases)

    # compute w point moving averates of daily cases and deaths
    wdcases = dcases.rolling(window=w,min_periods=1).mean()
    wddeaths = ddeaths.rolling(window=w,min_periods=1).mean()
    print('wdcases, w =',w,':')
    print(wdcases)
    print(wddeaths)



    '''
    sddeaths = pd.DataFrame(columns =['date','ddeaths'],index=[])
    sddeaths['date'] = all_dates
    sddeaths['ddeaths'] = ddeaths.stack().values
    print(sddeaths)
    print(sddeaths.sort_values(by='date',ascending=True))

    '''

    ratio = (wddeaths/wdcases).fillna(0.0)
#   print(ratio)
    max_ratio = ratio.max(axis=1)
    idmax_ratio = ratio.idxmax(axis=1)
#   print('max_ratio:')
#   print(max_ratio)
#   print('stacked:')
    sratio = pd.DataFrame(columns =['date','ratio'],index=[])
    sratio['date'] = all_dates
    sratio['ratio'] = ratio.stack().values
#   print(sratio)
#   if (1): sys.exit(1)

    ridge_file = 'CFR_ridge.csv'
    with open(ridge_file,'w') as rr:
        line = '{0: d}{1: d}{2: d}\n'.format(nG,w,ndate)
        rr.write(line)
        line = ''
        for k in range(0,ndate):
            line = line + ' {0}'.format(datetime.strftime(date_list[k],'%Y-%m-%d'))
        #   all_dates[k] = datetime.strftime(date_list[d],'%Y-%m-%d')#date_list[d]
        line = line + '\n'
        rr.write(line)

        line = ''
        for mr in idmax_ratio:
            line = line + '{0: d}'.format(mr)
        line = line + '\n'
        rr.write(line)

        line = ''
        for mr in max_ratio:
            line = line + '{0: .3f}'.format(mr)
        line = line + '\n'
        rr.write(line)
        sratio.to_csv(rr,index=True)
    print('CFR estimates for',nG,'geographies and',len(date_list),'dates written to',ridge_file)

def plot_CFR_ridge(CFRfile):

#with open('in.txt') as f:
#   data = []
#   cols,rows=list(map(int, f.readline().split()))
#   for i in range(0, rows):
#      data.append(list(map(int, f.readline().split()[:cols])))

    with open(CFRfile,'r') as rr:
        print('Reading',CFRfile)
        nG,window,ndate = (map(int, rr.readline().split()))
        print(nG, window, ndate)
        l1 = rr.readline()
        date_list = l1.split(None)
        l2 = rr.readline()
        idx_max = np.array(l2.split(sep=None), dtype=np.int64)
        l3 = rr.readline()
        rmax = np.array(l3.split(sep=None), dtype=np.float64)

        CFR = pd.read_csv(rr,header=0,index_col=0)
    #   print(CFR)
        
    print('finished reading',CFRfile,CFR.shape)
    print(CFR)

    meanCFR = CFR['ratio'].mean()
    print('mean CFR =',meanCFR)
    
    print('building labels')
    labels = [None]*ndate
#   prev_mon = 11
#   dlist = date_list[0].split('-')
#   prev_mon = int(dlist[1]) - 1
#   if prev_mon < 1: prev_mon=12
#   print(date_list[0],prev_mon)
    for i, d in enumerate(date_list):
        dlist = d.split('-')
        if (i == 0):
            prev_mon = int(dlist[1])
            dd = datetime.strptime(d,'%Y-%m-%d')
            labels[i] = dd.strftime('%Y')+' '+ dd.strftime('%b')
        elif (dlist[1] == '01' and prev_mon == 12):
            prev_mon = 1
            dd = datetime.strptime(d,'%Y-%m-%d')
            labels[i] = dd.strftime('%Y')+' '+ dd.strftime('%b')+'X'
        elif (dlist[2] == '01'): # day 1 of month
            prev_mon = int(dlist[1])
            dd = datetime.strptime(d,'%Y-%m-%d')
            labels[i] = dd.strftime('%b')+'Y'
        else:
                prev_mon = int(dlist[1])
    print(labels)    
#   if (1): sys.exit(1)

    print('plotting ridgeline ... very slowly')
    fig,axes = joyplot(CFR, by='date', column='ratio', labels = labels,
                       range_style='own', # tails = 0.2, 
             #         ylim='max',
                       overlap = 3, 
                       x_range=[0.0,0.06],
                       grid="y", linewidth=0.25, legend=False, figsize=(6.5,8.0),
                       title='Case Fatality Ratio\n'+str(nG) + ' Largest US Counties',
             #         colormap=cm.autumn_r)
                       colormap=cm.Blues_r)

    print(type(axes),len(axes),type(axes[0]))
    print('xlim: ',axes[0].get_xlim())
    print('ylim: ',axes[0].get_ylim())
    for a, ax in enumerate(axes):
        ax.plot([meanCFR,meanCFR],axes[0].get_ylim(),c='orange',linewidth=0.5)

    #   ax.text(rmax[a],ax.get_ylim()[1],str(a),c='red',ha='center')
    #   ax.axvline(meanCFR,c='orange',linewidth=0.5)
    #   ax.text(meanCFR,ax.get_ylim()[1],str(a),c='red',ha='left')
    #   ax.text(meanCFR,0.25*ax.get_ylim()[1],str(a),c='green',ha='right')
    #   ax.text(meanCFR,0.50*ax.get_ylim()[1],str(a),c='purple',ha='right')
    #   axes[a].text(meanCFR,1.0,str(a),c='purple',ha='right')

    gfile = 'CFRridgeline.png' 
    print('saving',gfile)
    plt.savefig(gfile, dpi=300)
    print('ridgeline plot saved as',gfile)
    plt.show()

# --------------------------------------------------       
print('------- here ------')
print('python:',sys.version)
print('pandas:',pd.__version__)
print('matplotib:',matplotlib.__version__)
print('joyplot:',joypy.__version__)

#_NOT_CFR_comp(3)
#CFR_comp(nG=30, w = 20)
plot_CFR_ridge('CFR_ridge.csv')

