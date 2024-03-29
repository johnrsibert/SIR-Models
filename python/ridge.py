#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jsibert
"""

import os
import sys
joypy_dir = '/home/jsibert/Projects/Python/joypy'
sys.path.append(joypy_dir)
from joypy import joyplot

from covid21 import config as cv
from covid21 import Geography as GG
#from covid21 import Fit as FF
#from covid21 import GraphicsUtilities as GU
#from covid21 import CFR

from numpy import errstate,isneginf #,array
import pandas as pd
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from matplotlib import rc
import numpy as np
#import pyreadr
#from io import StringIO 
#from io import BytesIO
#import base64
#import scipy.stats as stats
#import time
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

#    d1 = '2021-06-12'
#    d2 = '2021-06-18'

    d1 = '2020-04-01' # avoid axis label on 2020-02-29
    d2 = cv.nyt_county_dat['date'].iloc[-1] # last record
    date_list = pd.DatetimeIndex(pd.date_range(d1,d2),freq='D')
    print(len(date_list),date_list)
    ddate_list = date_list[1:]
    print(len(ddate_list),ddate_list)
#    if (1): sys.exit(1)
    print('processing',nG,'geographies and',len(date_list),'dates:')
    ndate = len(ddate_list)
    size = nG*ndate

#   create stacked date vector
    all_dates = pd.Series(['']*size)
    k2 = -1
    for g in range(0,nG):
        k1 = k2 + 1
        k2 = k1 + ndate -1
        print(g,k1,k2)
        for d in range(0,len(ddate_list)): #ndate):
            k = k1 + d
            all_dates[k] = datetime.strftime(ddate_list[d],'%Y-%m-%d')#date_list[d]

    
    print('all_dates:')
    print(all_dates)
    print('Computing',size,'CFR estimates for',nG,'geographies and',ndate,'dates')
    #if (1): sys.exit(1)

    # extract cumulative cases and deaths from NYT data
    # for first (ie largest) nG Geographies
    cases = pd.DataFrame(None,columns=np.arange(0,nG), index = date_list)
    deaths = pd.DataFrame(None,columns=np.arange(0,nG), index = date_list)
    gcols = pd.Series('',index = deaths.columns)
    gg = cv.GeoIndex
    for g in range(0,nG):    
#    for g in range(60,63):
        fips = gg['fips'].iloc[g]
        county = gg['county'].iloc[g]
        code = gg['code'].iloc[g]
        population = gg['population'].iloc[g]
        moniker = GG.set_moniker(county,code)
        sname = GG.short_name(moniker)
        gcols[g] = sname
        print(g,':',gg['county'].iloc[g] ,fips,sname,population)
        fips_mask = (dat['fips'] == fips) & (dat['fips']>0)
    #   fips_entries = fips_mask.value_counts()[1]
   
        gindex =  dat[fips_mask].index
        for k in gindex:
            tmp = dat.loc[k]
            kdate = tmp['date']
            if kdate in date_list:
                cases.loc[kdate,g]  = tmp['cases']
                deaths.loc[kdate,g] = tmp['deaths']
                
                if (kdate == '2021-11-03' and g == 61):
                    print('-----------',g,tmp)
                    if(1): sys.exit(0)

#   print(deaths)
#   print(cases)
#   print(gcols)
    print(len(gcols),len(gcols.unique()))

    # compute daily cases and deaths by first differences
    dcases = pd.DataFrame(None,columns=np.arange(0,nG), index = ddate_list)
    ddeaths = pd.DataFrame(None,columns=np.arange(0,nG), index = ddate_list)
    for g in range(0,nG):
        dcases[g]  = np.diff(cases[g])
        ddeaths[g] = np.diff(deaths[g])

    print('ddeaths:')
    
    print(ddeaths)
    print('dcases:')
    print(dcases)

#    if (1):sys.exit(1)

    # compute w point moving averates of daily cases and deaths
    wdcases = dcases.rolling(window=w,min_periods=1).mean()
    wddeaths = ddeaths.rolling(window=w,min_periods=1).mean()
    print('wdcases, w =',w,':')
    print(wdcases)
    print(wddeaths)

    ratio = wddeaths.divide(wdcases)
    print(ratio)
#   ratio = wddeaths/(wdcases+1e-8) #).fillna(0.0)
#   ratio = pd.DataFrame(wddeaths.divide(wdcases),index=wddeaths.index).fillna(0.0)
#   print(ratio.shape)
    ratio.replace([np.inf, -np.inf], [None, None], inplace=True)
    print(ratio)
#   df.loc[:, (df != 0).any(axis=0)] 
#    ratio = ratio.loc[:, (ratio > 0.0).any(axis=1)] 

    CFR_file = 'CFR.csv'
#    don't do this because there are duplicate names for larger nG    
#    ratio.columns = gcols
    print(ratio.shape)
    print('ratio:',type(ratio))
    print(ratio)
#    if (1): sys.exit(1)
    with open(CFR_file,'w') as rr:
        rr.write('# window\n')
        line = '{0: 8d}\n'.format(w)
        rr.write(line)
        ratio.to_csv(rr,index=True)

    print('CFR estimates for',nG,'geographies and',ndate,'dates written to',CFR_file)
#    if (1): sys.exit(1)

#    print(ratio.loc['2021-01-03'])
#   replece inf wieh NaN    
#    ratio.replace([np.inf, -np.inf], None, inplace=True)
#    print(ratio.loc['2021-01-03'])

    min_ratio = ratio.min(axis=1)
#    idmax_ratio = ratio.idxmax(axis=1)
    print('min_ratio:')
    print(min_ratio)
    max_ratio = ratio.max(axis=1)
    idmax_ratio = ratio.idxmax(axis=1)
    print('max_ratio:')
    print(max_ratio)
    print('stacked:')
    sratio = pd.Series(None,index=all_dates.index,dtype='object') # pd.DataFrame(None,columns =['date','ratio'])
    print(all_dates)
    sratio = ratio.stack()
    #CFR['ratio'] = ratio.stack()
    #CFR['date'] = all_dates
    #sratio.index = all_dates
    print(sratio)
    sratio = sratio.dropna()
    #sratio['date'] = pd.Series(ddate_list).stack()
    #sratio['ratio'] = ratio.stack().values
    print(sratio)
    # (1): sys.exit(1)

    ridge_file = 'CFR_ridge.csv'
    with open(ridge_file,'w') as rr:
        rr.write('# ndate   nG window\n')
        line = '{0: 7d}{1: 5d}{2: 7d}\n'.format(ndate,nG,w)
    #    line = '# window:\n{0: 5d}\ndate,ratio\n'.format(w)
        rr.write(line)
    
        sratio.to_csv(rr,header=True)
    print('CFR estimates for',nG,'geographies and',ndate,'dates written to',ridge_file)

def plot_CFR_ridge(CFRfile):
    with open(CFRfile,'r') as rr:
        print('Reading',CFRfile)
        line = '#'
        while line[0] == '#':
            line = rr.readline()
       
        ndate,nG,window = (map(int, line.split()))
        print(ndate, nG, window)
   
    #   l1 = rr.readline()
    #   date_list = pd.Series(l1.split(None))
    #   print(date_list)
    ###   l3 = rr.readline()
    #    rmax = np.array(l3.split(sep=None), dtype=np.float64)

        CFR = pd.read_csv(rr,header=0,comment='#') #index_col=0)
        CFR.columns=['date','G','ratio']
    #   print(CFR)
        
    print('finished reading',CFRfile,CFR.shape)
    print(CFR)
    CFR.drop(columns='G',inplace=True)
    #print(CFR)
       
    date_list = pd.Series(CFR['date'].unique())
    #print('date_list')
    #print(date_list)
    
#    CFR.replace(0, np.nan, inplace=True)
    zmask = CFR['ratio'] > 0

    print("CFR[zmask]['ratio']:")
#   print(CFR[zmask]['ratio'])
    print(CFR[zmask]['ratio'].mean())

    meanCFR = CFR['ratio'].mean()
    print('mean CFR =',meanCFR)
#   if (1): sys.exit(1)
    
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
            labels[i] = dd.strftime('%Y')+' '+ dd.strftime('%b')
        elif (dlist[2] == '01'): # day 1 of month
            prev_mon = int(dlist[1])
            dd = datetime.strptime(d,'%Y-%m-%d')
            labels[i] = dd.strftime('%b')
        else:
                prev_mon = int(dlist[1])
#   print(labels)    
#   if (1): sys.exit(1)

#   plt.rcParams['figure.constrained_layout.use'] = True
#              rcParams.update({"lines.linewidth": 2, ...}) 
    matplotlib.rcParams.update({"figure.autolayout": False})
    print('plotting ridgeline ... very slowly')
    fig,axes = joyplot(CFR[zmask], by='date', column='ratio', labels = labels,
                       range_style='own', # tails = 0.2, 
                       kind = 'lognorm',
                       ylim= 'own',#'max',
                       overlap = 4, 
                       x_range=[0.0,0.061],
                       grid="y", linewidth=0.25, legend=False, figsize=(6.5,8.0),
                       title='Case Fatality Ratio\n'+str(nG) + ' counties, ' + str(ndate) + ' days, ' + str(window) + ' day moving average',
             #          colormap=cm.autumn,
                       colormap=cm.Blues_r,
                       normalize = True, floc=None)

  #  print(type(axes),len(axes),type(axes[0]))#,len(rmax))
  #  print('xlim: ',axes[0].get_xlim())
  #  print('ylim: ',axes[0].get_ylim())
  #  for a, ax in enumerate(axes):
  #      ax.plot([meanCFR,meanCFR],ax.get_ylim(),c='orange',linewidth=2.0)
        
 #   ax = axes[len(axes)-2]
 #   print(ax.get_ylim())
 #   ax.plot([meanCFR,meanCFR],ax.get_ylim(),c='black',ls='--',
  #                         linewidth=0.5)
    

    gfile = cv.graphics_path+'CFRridge_{0:d}_{1:d}.png'.format(nG,window) 
    print('saving',gfile)
    plt.savefig(gfile, dpi=300)
    print('ridgeline plot saved as',gfile)
    plt.show()

# --------------------------------------------------       
print('------- here ------')
print('python:',sys.version)
print('pandas:',pd.__version__)
print('matplotib:',matplotlib.__version__)

#CFR_comp(nG=30, w = 23)

plot_CFR_ridge('CFR_ridge.csv')

