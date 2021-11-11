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
        line = '#'
        while line[0] == '#':
            line = rr.readline()
       
        window = int(line)
        print('window: ',window)
        CFR = pd.read_csv(rr,header=0,index_col=0,comment='#')
            
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
                                     # 'm','v','s','k'],
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
                #try:
                #    ss = pd.Series(stats.lognorm.stats(param.loc[d][0],param.loc[d][1],param.loc[d][2],
                #                               moments='mvsk'))
                #    print(ss)
                #    for i in range(0,4):
                #        param.loc[d][4+i] = ss[i]
                    
                #except Exception as e: 
                #    print('stats exception',e,' in row',row)
                #    print(param.loc[d])
       
                            
            except:
                fails += 1
                #print('fit exception in row',d,row)
                #print(param.loc[d])
                param.loc[d] = param.loc[d].fillna(1e-5)
                #print(param.loc[d])
       
                
       
    #    param.filna(0.0)   
        
        print('stats lognormal fit fialed for',fails,' dates')
        #print(param)
        
        fig, ax = plt.subplots(2,2,figsize=(11.0,6.5))
        Y = np.linspace(0.0,0.06,100)
   #     matplotlib.rcParams.update({"figure.autolayout": False})
        plt.rcParams["figure.autolayout"] = False
  
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
            
            ax[0,0].plot(Y,pdf,linewidth=1,color=colors[d])
            ax[0,0].set_ylabel('P(r)')
            ax[0,0].set_xlabel('Deaths/Cases (r)')
        
        ax[0,0].set_ylim(0.0,0.15)
   
        print(param)
            
     
   #     ax2 = ax[1,0].twinx()
   #     ax2.plot(mdate,np.log(param['scale']))
   #     GU.mark_ends(ax2, mdate,np.log(param['scale']) ,'mu')
   #     ax[1,0].plot(mdate,param['scale'])
   
        
        GU.make_date_axis(ax[1,0])
        ax[1,0].plot(mdate,param['scale'])
        GU.mark_ends(ax[1,0], mdate,param['scale'] ,'mu','b')
        ax[1,0].set_ylabel('Central Tendency')
        ax[1,0].plot(mdate,param['rmax'])
        GU.mark_ends(ax[1,0], mdate,param['rmax'] ,'mode','b')
        ax[1,0].set_ylim(0.0,0.06)
        #ax[1,0].plot(mdate,param['m'],linewidth=1)
        #GU.mark_ends(ax[1,0], mdate,param['m'] ,'m','r')
        #ax[1,0].set_ylim(0.0,0.06)
            
        GU.make_date_axis(ax[0,1])
        ax[0,1].plot(mdate,param['shape'])
        ax[0,1].set_ylabel('sigma = shape')
        #sv = param['v'].pow(0.5)
        #ax[0,1].plot(mdate,sv,linewidth=1)
        #GU.mark_ends(ax[1,0], mdate, sv,'s','r')
            
        GU.make_date_axis(ax[1,1])
        ax[1,1].plot(mdate,param['scale'])
        ax[1,1].set_ylabel('exp(mu) = scale')
        ax[1,1].set_ylim(0.0,0.06)
           
        #ax[1,1].plot(mdate,param['k'],linewidth=1)
        #GU.mark_ends(ax[1,1], mdate,param['k'] ,'k','r')
        
                                      
        title = 'CFR properties for {0:d} counties, {1:d} time periods, \
                 averaged over a {2:d} day moving window\n'.format(nG,ndate,window)
        #line = '{0: 7d}{1: 5d}{2: 7d}\n'.format(ndate,nG,w)
        #title = 'Title'
        print(title)
 
        #fig.suptitle(title,size='medium')
        plt.show()
        
        gfile = 'CFRtrend_'+str(ndate)+'_'+str(nG)+'_'+str(window)+'.png'
        plt.savefig(gfile,dpi=300)
        print('Plot saved as',gfile)
# -------------------------------

CFR_trends()