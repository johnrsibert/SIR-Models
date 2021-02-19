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
import scipy.stats as stats
import js_covid as cv
from covid21 import GraphicsUtilities as GU

def vline(ax, y, mark=None, pos = 'r'):
    ylim = ax.get_ylim()
    ax.plot((y,y), ylim)
    if (mark is not None):
        GU.mark_ends(ax, (y,y), ylim, mark, pos)



def plot_CFR_stats(date0=None):
    dat = pd.read_csv(cv.NYT_counties,header=0)
    dat['fips'] = dat['fips'].fillna(0).astype(np.int64) 
    dat.loc[dat['county'] == 'New York City','fips'] = 36999
    dat['deaths'] = dat['deaths'].fillna(0).astype(np.int64)
    
    if date0 is None:
       date0 = dat['date'][0]
 
    idx0 = dat['date'].searchsorted(date0)
    dat = dat[idx0:]    
    print('dat[idx0:]')    
    print(dat)
    print('last row:')
    most_recent = dat['date'].iloc[-1]
    print(most_recent)
    
    dmask = dat['deaths'] > 10.0
    cmask = dat['cases'] > 50.0
    date_mask = dat['date'] == most_recent
    zmask = cmask & dmask & date_mask
    dat = dat[zmask]
 #   dat = dat[date_mask]
    print(dat)
       
    print('computing CFR')
    ratio = dat['deaths']/dat['cases']
    dat['ratio'] = ratio
    print(dat)
 #        udates = pd.Series(dat['date'].unique())
    ufips = pd.Series(dat['fips'].unique())
    print('Using',len(ufips),'US counties')
   
    print('quantiles:')     
    qq = [0.95,0.975,0.99]
    for q in qq:
        print(q,np.quantile(dat['ratio'],q=q))
        
    mean = np.mean(dat['ratio'])
    std  = np.std(dat['ratio'])
    print('mean, std:', mean, std)
    meanlog = np.mean(np.log(dat['ratio']))
    stdlog  = np.std( np.log(dat['ratio']))
    
      
    # fit log normal distribution to UNtransformed data
    lnparam = stats.lognorm.fit(dat['ratio'],floc=0)
    print('lnparam =',lnparam)
    Y = np.linspace(0.0,0.08,100)
#    print('Y:',Y)
    # compute lognormal pdf from ln parameters
    lpdf =  stats.lognorm.pdf(Y,lnparam[0],lnparam[1],lnparam[2])
    lpdf = lpdf/lpdf.sum()
#    print('Y:',Y[0],Y[-1],len(Y))
   
 
    fig, ax = plt.subplots(2,figsize=(6.5,6.5))
    
    ax[0].set_ylabel('lognorm p(Y)')
    ax[0].set_ylim(0.0,1.1*lpdf.max())
    ax[0].set_xlim(Y[0],Y[-1])
    # plot log normal pdf
    ax[0].plot(Y,lpdf,label='lognormal fit',linewidth=5)
    ax[0].set_xlabel('Y')
    tx = ax[0].get_xlim()[1] #pd.Series(ax[0].get_xlim()).sum()/2
    ty = ax[0].get_ylim()[1]
    ax[0].text(tx,ty,'Untransformed Parameters',ha='right')

    lnmu = lnparam[2]
    vline(ax[0],lnmu, mark=' mean\n {: .4f}'.format(np.exp(meanlog))) #lnmu))
    
#    X = Y 
    X = np.linspace(-Y[-1], Y[-1],100)
#   X = np.logspace(-Y[-1], Y[-1],100)
#    print('X:',X)
    # fit normal distribution to log transformed data
    nparam = stats.norm.fit(np.log(dat['ratio']))
    print('nparam =',nparam)
    print('meanlog, stdlog =',meanlog,stdlog)
   
    # compute normal pdf from normal  parameters
    # norm.pdf(x, loc, scale)
    npdf= stats.norm.pdf(X, mean, std)#, std)
    npdf = npdf/npdf.sum()
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('normal p(X)')
    ax[1].set_ylim(0.0,1.1*npdf.max())
    ax[1].plot(X,npdf)
    tx = ax[1].get_xlim()[0] #pd.Series(ax[0].get_xlim()).sum()/2
    ty = ax[1].get_ylim()[1]
    ax[1].text(tx,ty,'Log-transformed Parameters',ha='left')

    vline(ax[1],mean,mark=' mean\n {: .4f}'.format(mean))
    
#   https://stackoverflow.com/questions/18534562/scipy-lognormal-fitting
    # compute lognormal pdf from mean and std of log transformed data
    lpdf1 = stats.lognorm.pdf(Y,stdlog,0.0,np.exp(meanlog))
    lpdf1 = lpdf1/lpdf1.sum()
    # overlay on original lognormal pdf
    ax[0].plot(Y,lpdf1,label='parameters from log(data)',linewidth=2)
#    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    ax[0].legend(loc='center right',frameon=False)



    if (len(ax) > 2):
        ax[2].plot(Y,X)


plot_CFR_stats()
