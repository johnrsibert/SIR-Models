#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 07:58:31 2021

@author: jsibert
"""
#from covid21 import config as cv
import pandas as pd
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc
from matplotlib import cm
import numpy as np
from joypy import joyplot
import js_covid as cv


def plot_CFR_ridge(date0=None):
    dat = pd.read_csv(cv.NYT_counties,header=0)
    dat['fips'] = dat['fips'].fillna(0).astype(np.int64) 
    dat.loc[dat['county'] == 'New York City','fips'] = 36999
    dat['deaths'] = dat['deaths'].fillna(0).astype(np.int64)
    
    if date0 is None:
       date0 = dat['date'][0]
 
    idx0 = dat['date'].searchsorted(date0)
    dat = dat[idx0:]    
    print(dat)
    
    dmask = dat['deaths'] > 0.0
    cmask = dat['cases'] > 0.0
    zmask = dmask & cmask
    dat = dat[cmask]
    print(dat)
    
    print('composing labels')
    udates = pd.Series(dat['date'].unique())
    labels = [None]*(len(udates))
    print(len(udates),'unique dates')
    prev_year = '1999'
    for i,d in enumerate(udates):
        dd = datetime.strptime(d,'%Y-%m-%d')
        if (i==0):
            prev_year = dd.strftime('%Y')
            labels[i] = dd.strftime('%Y')+' '+dd.strftime('%b')
        else:
            day = dd.strftime('%d')
            if (day == '01'):
                curr_year = dd.strftime('%Y')
                if (curr_year != prev_year):
                    prev_year = curr_year
                    labels[i] = dd.strftime('%Y')+' '+dd.strftime('%b')
                else:
                    labels[i] = dd.strftime('%b')
        if (labels[i] is not None):            
            print(i,d,labels[i])
    
    print('computing CFR')
    ratio = dat['deaths']/dat['cases']
    dat['ratio'] = ratio
    print(dat)
    
    print('quantiles:')     
    qq = [0.95,0.975,0.99]
    for q in qq:
        print(q,np.quantile(dat['ratio'],q=q))

    
    print('plotting ridgeline')
    fig,axes = joyplot(dat, by='date', column='ratio', labels = labels,
                           range_style = 'all', overlap = 2, x_range=[1e-8,0.08],
                           grid = True, linewidth = 0.25, legend = False, 
                           figsize = (6.5,6.5), # kind='counts',
                           title='Case Fatality Ratio', colormap=cm.Blues)
    naxes = len(axes)
#    ax = fig.add_subplot(-1,1,1) #93, 1, (1, 2))
#    ax.plot(ax.get_xlim(),ax.get_ylim(),color='orange')
    print(naxes,'y axes')

    gfile = 'tCFRridgeline'+'.png' 
    print('saving',gfile)
    plt.savefig(gfile, dpi=300)
    print('ridgeline plot saved as',gfile)
    plt.show()
    
plot_CFR_ridge() #'2020-03-19')   
