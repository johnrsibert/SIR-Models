#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 07:58:31 2021

@author: jsibert
"""

import pandas as pd
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc
from matplotlib import cm
import numpy as np
import scipy.stats as stats
#import js_covid as cv
from covid21 import GraphicsUtilities as GU
from covid21 import Geography as GG
from covid21 import config as cv


#def vline(ax, y, mark=None, pos = 'r'):
#    ylim = ax.get_ylim()
#    ax.plot((y,y), ylim)
#    if (mark is not None):
#        GU.mark_ends(ax, (y,y), ylim, mark, pos)

def vline(ax, x, label=None, ylim=None, pos='center'):
    if ylim is None:
        ylim = ax.get_ylim()
    ax.plot((x,x), ylim, linewidth=1, linestyle=':')
    c = ax.get_lines()[-1].get_color()
    ax.text(x,ylim[1], label, ha=pos,va='bottom', linespacing=1.8,
            fontsize=8, color=c)

def save_plot(plt,save,n,what):
    if save:
        gfile = cv.graphics_path+'CFR_'+what+'_'+str(n)+'.png'
        plt.savefig(gfile,dpi=300)
        plt.show(block=False)
    #   plt.pause(5)
        print('Plot saved as',gfile)
    else:
        plt.show()
    plt.close()

def plot_CFR_lognorm_fit(date0=None):
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
    
    dmask = dat['deaths'] > 1.0
    cmask = dat['cases'] > 1.0
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
    npdf= stats.norm.pdf(X, mean, std)#, std)   dat = pd.read_csv(cv.NYT_counties,header=0)
    dat['fips'] = dat['fips'].fillna(0).astype(np.int64) 
    dat.loc[dat['county'] == 'New York City','fips'] = 36999
    dat['deaths'] = dat['deaths'].fillna(0).astype(np.int64)
   
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
        
def plot_recent_CFR(save = False):
    dat = pd.read_csv(cv.NYT_counties,header=0)
    dat['fips'] = dat['fips'].fillna(0).astype(np.int64) 
    dat.loc[dat['county'] == 'New York City','fips'] = 36999
    dat['deaths'] = dat['deaths'].fillna(0).astype(np.int64)
    most_recent = dat['date'].iloc[-1]
    print('most recent date:',most_recent)
    
    dmask = dat['deaths'] > 1.0
    cmask = dat['cases'] > 1.0
    date_mask = dat['date'] == most_recent
    fmask = dat['fips']  > 0
    zmask = cmask & dmask & date_mask & fmask
    dat = dat[zmask]
 #   dat = dat[date_mask]
    print(dat)
       
    print('computing CFR')
    ratio = dat['deaths']/dat['cases']
    dat['cfr'] = ratio
    print(dat)
    ufips = len(pd.Series(dat['fips'].unique()))
    print('Using',ufips,'US counties')
    
    
    fig, ax = plt.subplots(1,figsize=(6.5,4.5))
    bins = np.linspace(0.0,0.08,81)
    print(bins)
#    nbins = len(bins)
#    print(nbins)
    xticks = np.linspace(0.0,0.08,num=9)
    ax.set_xlim(xticks[0],xticks[len(xticks)-1])
    ax.set_xlabel('Most Recent Case Fatality Ratio')
    ax.set_ylabel('Number')
    ax.set_xticks(xticks)
#   hist,edges,patches = ax.hist(recent['cfr'],nbins)
    weights = np.ones_like(dat['cfr']) / len(dat['cfr'])
    hist,edges,patches = ax.hist(dat['cfr'],weights=weights,bins=bins,
                                 density=True)

    param = stats.lognorm.fit(dat['cfr'],floc=0)
    pdf = stats.lognorm.pdf(edges,param[0],param[1],param[2])
    print('param:',param)
    hb = 0.5*(bins[1]-bins[0])
    ax.plot(bins+hb,pdf,linewidth=1,linestyle='--')

    mean = np.mean(dat['cfr'])+hb
#   meanlog = np.mean(np.log(dat['cfr']))
#   print('meanlog =',meanlog)
    std = np.std(dat['cfr'])
    print('mean, std:', mean, std)
#   median = np.median(dat['cfr'])+hb
    Q95 = np.quantile(dat['cfr'],q=0.975)+hb
    mode = bins[pd.Series(pdf).idxmax()]+hb
    ylim = ax.get_ylim()
    ylim = (ylim[0],0.95*ylim[1])
#   vline(ax,median,'Median',ylim=ylim,pos='left')
    note = ' Mean\n {: .4f}'.format(mean)
    vline(ax,mean,note,pos='left')
   
    note = ' Mode\n {: .4f}'.format(mode)
    vline(ax,mode, note,pos='right')
    vline(ax,Q95,' 97.5%')
    
    xlim = ax.get_xlim()
#   tx = xlim[0]+0.95*(xlim[1]-xlim[0])
    tx = xlim[1]
    ylim = ax.get_ylim()
    tcases = int(dat['cases'].sum())
    tdeaths = int(dat['deaths'].sum())
    ty = ylim[0]+0.90*(ylim[1]-ylim[0])
    note = '{0} Counties; {1:,} Cases; {2:,} Deaths'.format(ufips,tcases,tdeaths)
    ax.text(tx,ty,note,ha='right',va='center',fontsize=10)
    GU.add_data_source(fig)

    save_plot(plt,save,'all_recent','hist')
    
    
def plot_DC_scatter(nG=5, save=False):
    
    def set_axes(ax):
        ax.set_yscale('log')
        ax.set_ylabel('Deaths')
        ax.set_ylim(1,1e6)
        ax.set_xscale('log')
        ax.set_xlabel('Cases')
        ax.set_xlim(10,1e7)
    
    def mark_points(coll,ax,x,y,label,end='b',spacer=' '):
        c = coll.get_facecolor()[0]
        if ( (end =='l') | (end == 'b')):
            mark = ax.text(x[0],y[0],label+spacer,ha='right',va='center',fontsize=8,
                    color=c)

        if ( (end =='r') | (end == 'b')):
            i = len(x)-1
            mark = ax.text(x.iloc[i],y.iloc[i],spacer+label,ha='left',va='center',fontsize=8,
                    color=c)
           
    def plot_cmr(a, rr=[2.0]):
        for r in rr:
            rstr = str(r)
            xr = [0.0]*2
            yr = [0.0]*2
            for i in range(0,len(yr)):
                xr[i] = a.get_xlim()[i]
                yr[i] = xr[i]*r/100.0

            a.plot(xr,yr, linewidth=1,color='0.1',alpha=0.5)  
            GU.mark_ends(a,xr,yr,rstr,'r',' ')
            
    def save_plot(plt,save,n,what):
        if save:
            gfile = cv.graphics_path+'xCFR_'+what+'_'+str(n)+'.png'
            plt.savefig(gfile,dpi=300)
            plt.show(block=False)
            plt.pause(5)
            
            print('Plot saved as',gfile)
        else:
            plt.show()
        plt.close()
 
#   read NYT data; set NYC fips code correct type of deaths column    
    dat = pd.read_csv(cv.NYT_counties,header=0)
    dat['fips'] = dat['fips'].fillna(0).astype(np.int64) 
    dat.loc[dat['county'] == 'New York City','fips'] = 36999
    dat['deaths'] = dat['deaths'].fillna(0).astype(np.int64)
#    print(dat.columns)

#   get the geography index; assume sorted by decreasing population
    glist = pd.read_csv(cv.GeoIndexPath,header=0,comment='#')
#    print(glist.columns)

#   set some graphics defaults for scatter plots    
    plt.rcParams["scatter.marker"] = '.'
    plt.rcParams["lines.markersize"] = 3

#   mask for the most populous nG geographies
#     county,state,fips,code,flag,population
#1    Los Angeles,California,6037,CA,sm,10039107
#2    New York City,New York,36999,NY,m,8336817   
#3    Cook,Illinois,17031,IL,m,5150233
#4    Harris,Texas,48201,TX,m,4713325
#5    Maricopa,Arizona,4013,AZ,m,4485414
#6    San Diego,California,6073,CA,sm,3338330
#7    Orange,California,6059,CA,m,3175692
#8    Miami-Dade,Florida,12086,FL,m,2716940#    print(nG,glist['population'].iloc[nG])

    min_pop = glist['population'].iloc[nG]                 
    pop_mask = glist['population'] > min_pop #4485413
    gg = glist[pop_mask]

    fig, ax = plt.subplots(1,figsize=(6.5,4.5))
    set_axes(ax)
    
#    nG = len(gg)
    for g in range(nG):
     #   print(g,gg.iloc[g])
     #   cmask = dat['county'] == gg['county'].iloc[g]
        fmask = dat['fips'] == gg['fips'].iloc[g]
     #   zmask = cmask & fmask
        cc = dat[fmask]
#        print(cc.tail())
        coll = ax.scatter(cc['cases'],cc['deaths'])
        sn = GG.short_name(GG.set_moniker(gg['county'].iloc[g],gg['code'].iloc[g]))
        mark_points(coll,ax,cc['cases'],cc['deaths'],sn,'r')
    
    plot_cmr(ax, [0.5,1.0,2.0,4.0,8.0])
    logxlim = np.log(ax.get_xlim())
    logylim = np.log(ax.get_ylim())
    tx = np.exp(logxlim[0]+0.05*(logxlim[1]-logxlim[0]))
    ty = np.exp(logylim[0]+0.95*(logylim[1]-logylim[0]))
    # cases and deaths sums too large, so omit for now
#    note = '{0} Largest Counties; {1:,} Cases; {2:,} Deaths'.format(nG,cc['cases'].sum(),cc['deaths'].sum())
    note = '{0} Largest Counties'.format(nG)
    ax.text(tx,ty,note ,ha='left',va='center',fontsize=10)
    GU.add_data_source(fig)  
    save_plot(plt,save,nG,'all')

 
#   mask to exlude records with 'Unknown' counties
    umask = dat['county'] != 'Unknown'  
    cc = dat[umask]
#    print(cc)
    
    ufips = pd.Series(cc['fips'].unique())
#    print(ufips)
    
    fig, ax = plt.subplots(1,figsize=(6.5,4.5))
    set_axes(ax)

    kG = len(ufips)
    for g in range(kG):
        fmask = cc['fips'] == ufips.iloc[g]
    #    print(ufips.iloc[g],end="\r", flush=True)
    #    print(cc[fmask])
        ax.scatter(cc[fmask]['cases'],cc[fmask]['deaths'])
        
    logxlim = np.log(ax.get_xlim())
    logylim = np.log(ax.get_ylim())
    tx = np.exp(logxlim[0]+0.05*(logxlim[1]-logxlim[0]))
    ty = np.exp(logylim[0]+0.95*(logylim[1]-logylim[0]))
    # cases and deaths sums too large, so omit for now
#    note = '{0} Largest Counties; {1:,} Cases; {2:,} Deaths'.format(nG,cc['cases'].sum(),cc['deaths'].sum())
    note = '{0} "Known" Counties'.format(kG)
    ax.text(tx,ty,note ,ha='left',va='center',fontsize=10)
    GU.add_data_source(fig)  
    save_plot(plt,save,'0000','all')

plot_DC_scatter(save=True) #nG=6,save=True)
#plot_recent_CFR(save=True)
#plot_CFR_lognorm_fit()
