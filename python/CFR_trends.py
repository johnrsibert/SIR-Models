#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 08:27:21 2021

@author: jsibert
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.style.use('file:///home/jsibert/.config/matplotlib/john.mplstyle')
import numpy as np
import scipy.stats as stats
import pandas as pd
import sys
from covid21 import GraphicsUtilities as GU
#import scipy
#print('scipy version:',scipy.__version__)
'''
https://stackoverflow.com/questions/8747761/scipy-lognormal-distribution-parameters
'''
def CFR_trends(CFRfile = 'CFR.csv'):

    with open(CFRfile,'r') as rr:
        print('Reading',CFRfile)
        
    #    window = None
    #   nG,window,ndate = (map(int, rr.readline().split()))
    #    window = map(int, rr.readline().split())
    #    wl = rr.readline()
    #    print(wl)
    #    window = int(wl)
    #   this only works if there is not 'comment' line at head of thhe file
        window = int(rr.readline())
         
        print('window: ',window)
        CFR = pd.read_csv(rr,header=1,index_col=0,comment='#')
            
    print('finished reading',CFRfile,CFR.shape)
    
    nG = CFR.shape[1]
    ndate = CFR.shape[0]       
    print('nG, ndate:',nG, ndate)
    print(CFR)
#    CFR.replace([np.inf, -np.inf], None, inplace=True)
#    if (1):sys.exit(1)
#    print(CFR.min(axis=0))
#    print(CFR.max(axis=0))
#    CFR = np.pos(CFR)
#    print(CFR)
#    print(CFR.min(axis=0))
#    print(CFR.max(axis=0))  
 
   
    all_dates = True
    if not all_dates:
        dat = CFR.iloc[120]
               
        lnparam = stats.lognorm.fit(dat,floc=0)
        print('lnparam =',lnparam)
        Y = np.linspace(0.0,0.08,100)
    #    print('Y:',Y)
        # compute lognormal pdf from ln parameters
        lpdf =  stats.lognorm.pdf(Y,lnparam[0],lnparam[1],lnparam[2])
        lpdf = lpdf/lpdf.sum()
        fig, ax = plt.subplots(1)
    #    print('Y:',Y[0],Y[-1],len(Y))
       
        ax.set_ylabel('lognorm p(Y)')
        ax.set_ylim(0.0,1.1*lpdf.max())
        ax.set_xlim(Y[0],Y[-1])
        # plot log normal pdf
        ax.plot(Y,lpdf,label='lognormal fit',linewidth=5)
        ax.set_xlabel('Y')
        
        meanlog = np.log(dat).mean()
        ax.axvline(np.exp(meanlog),linewidth=1)
        plt.show()
   
    else:
        fails = 0
        param = pd.DataFrame(None,
                             columns=['shape','loc','scale','rmax'],
                             index=CFR.index)
        for d in CFR.index:
        #    print(d)
            row = CFR.loc[d]
            mask = row > 0.0
        #    print(mask)
        #    print(row)
        
        #   lnparam      = stats.lognorm.fit(dat,floc=0)
            try:
                param.loc[d][0:3] = stats.lognorm.fit(row[mask].dropna(),
                                                 floc=0)
            
            except:
                print('fit exception in row',d,row)
                #print(param.loc[d])
                param.loc[d] = param.loc[d].fillna(1e-5)
                #print(param.loc[d])
       
                
       
    #    param.filna(0.0)   
    #    print(param)
        matplotlib.rcParams.update({"figure.autolayout": False})
        fig, ax = plt.subplots(2,2,figsize=(11.0,6.5))
        Y = np.linspace(0.0,0.06,100)
 
        colors = plt.cm.Blues_r(np.linspace(0,1,ndate))
   #     colors = plt.cm.viridis(np.linspace(0,1,ndate))
        mdate = mdates.date2num(param.index)
        for d in range(0,len(param)):
            p = param.iloc[d]
        #    print(p)
            pdf = pd.Series(stats.lognorm.pdf(Y,p['shape'],p['loc'],p['scale']))
        #    cdf = pd.Series(stats.lognorm.cdf(Y,p['shape'],p['loc'],p['scale']))
            pdf = pdf/pdf.sum()
        
        #   find mode, the value of Y where pdf is maximum
            rmax = 0.0
            for i in range(0,len(pdf)-1):
                if pdf[i] > pdf[i+1]:
                    rmax = Y[i] + 0.5*(Y[i]-Y[i+1])
                    param.iloc[d]['rmax'] = rmax
                #    print('rmax =',rmax, 'at ',i)
                    break
            
            ax[0,0].plot(Y,pdf,linewidth=2,color=colors[d])
            ax[0,0].set_ylabel('P(r)')
            ax[0,0].set_xlabel('Deaths/Cases (r)')

    #    print(param)
            
        GU.make_date_axis(ax[1,0])
    #    ax[1,0].set_ylabel('central tendencies)')
        '''
df = pd.DataFrame({'one': range(10), 'two': range(10, 20)})

ax = df['one'].plot()
ax2 = df['two'].plot(secondary_y=True)
        '''
   #     ax2 = ax[1,0].twinx()
   #     ax2.plot(mdate,np.log(param['scale']))
   #     GU.mark_ends(ax2, mdate,np.log(param['scale']) ,'mu')
   #     ax[1,0].plot(mdate,param['scale'])
   
        ax[0,0].set_ylim(0.0,0.15)
   
        ax[1,0].plot(mdate,param['scale'])
        GU.mark_ends(ax[1,0], mdate,param['scale'] ,'mu','r')
        ax[1,0].set_ylabel('Central Tendency')
   #    ax[1,0].plot((0,0),(0,0),linewidth=0.5)
        ax[1,0].plot(mdate,param['rmax'])
        GU.mark_ends(ax[1,0], mdate,param['rmax'] ,'mode','r')
        ax[1,0].set_ylim(0.0,0.06)
            
        GU.make_date_axis(ax[0,1])
        ax[0,1].plot(mdate,param['shape'])
        ax[0,1].set_ylabel('sigma = shape')
            
        GU.make_date_axis(ax[1,1])
        ax[1,1].plot(mdate,param['scale'])
        ax[1,1].set_ylabel('exp(mu) = scale')
        ax[1,1].set_ylim(0.0,0.06)
    
        plt.show()
        
        gfile = 'CFRtrend_'+str(ndate)+'_'+str(nG)+'_'+str(window)+'.png'
        plt.savefig(gfile,dpi=300)
        print('Plot saved as',gfile)
# -------------------------------

CFR_trends()