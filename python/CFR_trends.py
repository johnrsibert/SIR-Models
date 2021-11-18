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
from datetime import date, datetime, timedelta
from covid21 import config as cv
from covid21 import Geography as GG
from covid21 import GraphicsUtilities as GU
#import scipy
#print('scipy version:',scipy.__version__)

def CF_xcorr(nG=5,max_lag=21):
    dat = cv.nyt_county_dat 
    dat['fips'] = dat['fips'].fillna(0).astype(np.int64)
    NYC_mask = dat['county'] == 'New York City' 
    dat.loc[NYC_mask,'fips'] = 36999

    # creat vector of dates from first reported death untill ...
    k = 0
    while dat['deaths'][k]< 1.0:
        k += 1
#   d1 = cv.nyt_county_dat['date'].iloc[k] # first death
#    d1 = '2021-01-01'
#    d2 = '2021-02-01'
    d1 = '2020-04-01' 
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
#    print('Computing',size,'CFR estimates for',nG,'geographies and',ndate,'dates')
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

    print('deaths:',deaths)
    print('cases:',cases)


    for d in range(0,ndate):
#   for lag in range (1,max_lag+1):
        for lag in range (0,2):

#   print('lag =',lag,'Spearman r correletions and p-values for Triptan:')
    #    rs = '{:>4d}'.format(lag)
    #   a = metrics.iloc[lag:]['Triptan']
            a = deaths.iloc[d]
            print('a:',a)
    #    rs = rs +  '{:>5d} '.format(len(a))
    #    ds= '{:>10s}'.format('')
    #   b = metrics.iloc[:-lag,[i]]
            b = cases.iloc[d+lag]
            print('b:',b)
        #    print('b:',b)
        #   rr.iloc[lag-1][vnames[i]],pp.iloc[lag-1][vnames[i]], = stats.spearmanr(a, b)
        #   rr[vnames[i]],pp[vnames[i]] = stats.spearmanr(a, b)
            r,p =  stats.spearmanr(a, b)
            print('r:',r)
            print('p:',p)
    #    rs = rs + '  {:4.3f} '.format(r)
    #    ds = ds + ' ({:4.3f})'.format(r)
        
#    print(rs)
#    print(ds)    

'''
https://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas



53

As far as I can tell, there isn't a built in method that does exactly what you are asking. But if you look at the source code for the pandas Series method autocorr, you can see you've got the right idea:

def autocorr(self, lag=1):
    """
    Lag-N autocorrelation

    Parameters
    ----------
    lag : int, default 1
        Number of lags to apply before performing autocorrelation.

    Returns
    -------
    autocorr : float
    """
    return self.corr(self.shift(lag))

So a simple timelagged cross covariance function would be

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

Then if you wanted to look at the cross correlations at each month, you could do

 xcov_monthly = [crosscorr(datax, datay, lag=i) for i in range(12)]



'''



        



def CFR_trends(CFRfile = 'CFR.csv'):
    '''
    https://stackoverflow.com/questions/8747761/scipy-lognormal-distribution-parameters
    '''
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
   #    set rcParams["figure.autolayout"] (default: False) to True.
   #    plt.rcParams["figure.autolayout"] = False
        fig.set_tight_layout(False)
  
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
   
   #    usimg mark+ends (..., 'b') may result in error if an end is inf      
        GU.make_date_axis(ax[1,0])
        ax[1,0].plot(mdate,param['scale'])
        GU.mark_ends(ax[1,0], mdate,param['scale'] ,'mu','r')
        ax[1,0].set_ylabel('Central Tendency')
        ax[1,0].plot(mdate,param['rmax'])
        GU.mark_ends(ax[1,0], mdate,param['rmax'] ,'mode','r')
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
        
        fig.show()
                                     
        title = 'CFR properties for {0:d} counties, {1:d} time periods, averaged over a {2:d} day moving window\n'.format(nG,ndate,window)
        #line = '{0: 7d}{1: 5d}{2: 7d}\n'.format(ndate,nG,w)
        #title = 'Title'
        print(title)
 
        fig.suptitle(title,size='medium',y=1.05)
        
        gfile = 'CFRtrend_'+str(ndate)+'_'+str(nG)+'_'+str(window)+'.png'
        fig.savefig(gfile,dpi=300, bbox_inches='tight', pad_inches=0.25)
    #    fig.savefig('temp.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
        print('Plot saved as',gfile)

# -------------------------------

#CFR_trends()
CF_xcorr()