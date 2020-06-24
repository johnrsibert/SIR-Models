#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jsibert
"""

import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc
import numpy as np
import os
import pyreadr
from io import StringIO
import scipy.stats as stats
from sigfig import round
from tabulate import tabulate
from collections import OrderedDict

plt.style.use('file:///home/jsibert/.config/matplotlib/john.mplstyle')

LargestCACounties = ['Los Angeles', 'San Diego', 'Orange', 'Riverside',
                         'San Bernardino', 'Santa Clara', 'Alameda',
                         'Sacramento', 'Contra Costa']


largest_us_counties = ["AlamedaCA", "BexarTX", "BrowardFL", "ClarkNV", 
"Contra_CostaCA", "CookIL", "DallasTX", "FresnoCA", "HarrisTX", "KernCA",
"KingWA", "Los_AngelesCA", "MaricopaAZ", "MarinCA", "Miami-DadeFL",
"New_York_CityNY", "OrangeCA", "RiversideCA", "SacramentoCA",
"San_BernardinoCA", "San_DiegoCA", "San_FranciscoCA", "San_JoaquinCA",
"San_MateoCA", "Santa_ClaraCA", "SonomaCA", "StanislausCA",
"SuffolkMA", "TarrantTX", "VenturaCA", "WayneMI"]


NewYorkCounties = ['New York City']#,'Tompkins']
EastBayCounties = ['Santa Clara','Alameda','Contra Costa','Sonoma','Napa']
BayAreaCounties = EastBayCounties + ['San Mateo','San Francisco','Marin']

#big_county_list = ["New York City","Los Angeles","San Diego",
big_county_list = ["Los Angeles","San Diego",
                       "Orange", "Riverside",
                       "San Bernardino","Santa Clara", "Alameda",
                       "Sacramento","Contra Costa","Fresno", "Kern",
                       "San Francisco",
                       "Ventura","San Mateo","San Joaquin",
                       "Stanislaus","Sonoma","Marin"]

counties_path = "../us-counties.csv"
fit_path = '/home/jsibert/Projects/SIR-Models/fits/'
pop_data_path = '../co-est2019-pop.csv'

FirstNYTDate = datetime.strptime('2020-01-21','%Y-%m-%d')
CAOrderDate = datetime.strptime('2020-03-19','%Y-%m-%d')
EndOfTime = datetime.strptime('2020-06-30','%Y-%m-%d')

county_dat = pd.read_csv(counties_path,header=0)
#county_state = pd.read_csv('../county-state.csv',header=0)

#print(county_state.shape)
#print(county_state)
#for c in range(0,len(county_state)):
#    print('c:',c,county_state.loc[c])


def Strptime(x):
   """
   wrapper for datetime.strptime callable by map(..)
   """
   s = str(x)
   y = datetime.strptime(x,'%Y-%m-%d')
   return(y)

def prop_scale(lim,prop):
    s = lim[0] + prop*(lim[1]-lim[0])
    return(s)

def moving_average(a, n=3) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def median(x):
    mx = x.quantile(q=0.5)
    return float(mx)

def get_county_pop(County,State='California'):
    """
    Reads US Census populations estimates for 2019
    Subset of orginal csv file saved in UTF-8 format. 
    The word 'County' stipped from county names.

    nyc_pop is the sum of census estimates for New York, Kings, Queens
    Bronx, and Richmond counties to be compatible with NYTimes data
    """
    nyc_pop = 26161672
    if (County == 'New York City'):
        return(nyc_pop)

    dat = pd.read_csv(pop_data_path,header=0)
    state_filter = dat['state'].isin([State])
    county_filter = dat['county'].isin([County])
    County_rows = state_filter & county_filter
#   w = -1
#   for k in range(0,len(County_rows)):
#       if (County_rows[k]):
#           w=k
#   print(w)
#   print(County_rows[w])
    population = pd.to_numeric(dat[County_rows]['population'])
    return(population)


#def check_delta_t(dat,County):
#    State = 'California'
#    state_filter = dat['state'].isin([State])
#    county_filter = dat['county'].isin([County])
#    County_rows = state_filter & county_filter
#    cdat = dat[County_rows]
#
#    Date = cdat['date'].map(Strptime)
#    dDate = Date.diff()
#    n = len(dDate) - 1
#    print('Maximum delta t for',County,State,':',max(dDate[-n:]))
#    return(max(dDate[-n:]).days)

def short_name(s):
    w = s.split(' ')
#   print(len(w),w)
    if (len(w)<2):
        sn = s[0:3]
    else:
        sn = w[0][0]+w[1][0:2]
#   print(sn)
    return(sn)  

        
def mark_ends(ax,x,y,label,end='b',spacer=' '):
    sl = short_name(label)
    if ( (end =='l') | (end == 'b')):
        ax.text(x[0],y[0],sl+spacer,ha='right',va='center',fontsize=10)
   
    if ( (end =='r') | (end == 'b')):
        i = len(x)-1
        ax.text(x[i],y[i],spacer+sl,ha='left',va='center',fontsize=10)

def update_dat():
    os.system('git -C /home/other/nytimes-covid-19-data pull -v')
    make_dat_file()
    plot_county_dat(county_dat,County='Alameda',State='California',file='AlamedaCA')
#   plot_county_dat(county_dat,County='Marin',State='California',file='MarinCA')
    plot_county_dat(county_dat,County='Sonoma',State='California',file='SonomaCA')

def make_dat_file (cspath='../county-state.csv'):
    """ Generate input data for analysis by ADMB or TMB models
        dat: pandas object read by another function
        County: string containing the name of the county in California
    """

#   maxdt = check_delta_t(dat,County)
#   if (maxdt > 1):
#       print('WARNING: timstep greater than one for',County,'County')

    
#   pops = pd.read_csv(pop_data_path,header=0)        # get county population
#   cp_row = pops['county'] == County 
    
    csdat = pd.read_csv(cspath,header=0)

    mtime = os.path.getmtime("/home/other/nytimes-covid-19-data/us-counties.csv")
    dtime = datetime.fromtimestamp(mtime)
    update_stamp = str(dtime.date())
    dat = county_dat
#   print('nyt columns:',dat.columns)
#   print('cs columns:',csdat.columns)

    for cs in range(0,len(csdat)):
    #   print(csdat['county'][cs],csdat['state'][cs])
        population = get_county_pop(csdat['county'][cs],csdat['state'][cs])
    #   print(csdat['county'][cs],csdat['state'][cs],population)
    #   ppop = population
    #   print(ppop)#,population)


        State = csdat['state'][cs]
        County = csdat['county'][cs]
        ST = csdat['ST'][cs]
        state_filter = dat['state'].isin([State])
        county_filter = dat['county'].isin([County])
        County_rows = state_filter & county_filter

        Date = dat[County_rows]['date'].map(Strptime)
        dDate = Date.diff()
        n = len(dDate) - 1

        if (n < 1):
            print('* * * No records found for',County,State)
            return(0)
  
        print(len(Date),'records found for',County,State)
        print('    delta t:(',min(dDate[-n:]),'),(',max(dDate[-n:]),')')
   #    if (max(dDate[-n:])>1):
   #        print('* * * Missing time steps for',County,State)

        dfile = County.replace(' ','_',5)+ST+'.dat'
   #    print(County, ST,dfile)
        O = open(dfile,'w')
   #    O.write('# county\n %s\n'%(County.replace('_',' ',5)+ST))
        O.write('# county\n %s\n'%(County.replace(' ','_',5)+ST))
        O.write('# updateed from https://github.com/nytimes/covid-19-data.git\n')
        O.write(' %s\n'%update_stamp)
        O.write('# population (N0)\n %10d\n'%population)
   #    print(dat[County_rows])
        r0 = dat['date'][County_rows].index[0]
        date0 = dat['date'][r0]
    #   print('date0',date0)
        O.write('# date zero\n %s\n'%date0)
    #   print(dat[County_rows])
        ntime = len(dat[County_rows])-1
    #   print('----------------ntime',ntime)
        O.write('# ntime\n %4d\n'%ntime)
    #   print(dat[County_rows].index)
    #   print('date','index','cases','deaths')
        O.write('#%6s %5s\n'%('cases','deaths'))
        for r in dat[County_rows].index:
            O.write('%7d %6d\n'%(dat['cases'][r],dat['deaths'][r]))



def make_t0_series(dat,County,Column,threshold,State='California'):
    """ Generate a time series with time measured in days since a threshold was eqalled or surpassed
        dat: pandas object read by another function
        County: string containing the name of the county
        Column: string inentifying data needed; 'cases' of 'deaths'
        threshold: minimum number to consider
        State: state in which county belongs; eg 'California'
    """
    state_filter = dat['state'].isin([State])
    county_filter = dat['county'].isin([County])
    threshold_filter = dat[Column] > threshold
    County_rows = state_filter & county_filter & threshold_filter
    t0series = pd.DataFrame(columns=['time',Column],dtype=float)
    t0series['time'] = dat['Date'][County_rows]
    t0series[Column] =  dat[Column][County_rows]
    t0 = t0series['time'].iloc[0]
    t0series['time'] = t0series['time'] - t0
    t0series['time'] = pd.to_numeric(t0series['time'].dt.days, downcast='integer')

    return(t0series)

def read_ADMB_rep(rep_file = 'simpleSIR3.rep', meta_rows=9, ests_rows=8):
    rep_path = '/home/jsibert/Projects/SIR-Models/ADMB/'
    with  open(rep_path+rep_file) as rep:
        rep_text = rep.read()
    
    meta_pos = rep_text.find('# meta:')
    diag_pos = rep_text.find('# diag:')
    ests_pos = rep_text.find('# ests:')

    meta = pd.read_csv(StringIO(rep_text[meta_pos:]),comment='#',nrows=meta_rows,
                   lineterminator='\n',sep=',',index_col=False)
    ntime=int(get_metadata('ntime',meta))
#   print('meta:\n',meta)
#   print(ntime)
    diag = pd.read_csv(StringIO(rep_text[diag_pos:]),comment='#',nrows=ntime+1,
                       lineterminator='\n',sep=',',index_col=False)
#   print('diag:\n',diag)
    ests = pd.read_csv(StringIO(rep_text[ests_pos:]),comment='#',nrows=ests_rows,
                       lineterminator='\n',sep=',',index_col=False)
#   print('ests:\n',ests)


    fit = OrderedDict() # mimic pyreadr here
    fit['meta']=meta
    fit['diag'] = diag
    fit['ests'] = ests

    return(fit)


def plot_county_fit(county,fit_type = 'TMB',
                  yscale='log', per_capita=False, delta_ts=False,
                  text_spacer='  ', save = False):
    """ 
    Plot predicted & observed cases and deaths vs time from threshold
    Counties: list of California counties to plotted
    """

    firstDate = mdates.date2num(FirstNYTDate)
    orderDate = mdates.date2num(CAOrderDate)
    lastDate  = mdates.date2num(EndOfTime)

    date_list = pd.date_range(start=firstDate,end=lastDate)
    date_lim = [date_list[0],date_list[len(date_list)-1]]
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams["scatter.marker"] = '+'
    npl = 3
    fig, ax = plt.subplots(npl,1,figsize=(6.5,npl*2.5))
    if (per_capita):
        ax[0].set_ylabel('Cases'+' per '+str(mult))
        ax[1].set_ylabel('Deaths'+' per '+str(mult))
    else:
        ax[0].set_ylabel('Cases')
        ax[1].set_ylabel('Deaths')
    if (npl > 2):
        ax[2].set_ylabel(r'$\beta\ (da^{-1})$')


    for a in range(0,len(ax)):
        ax[a].set_xlim([firstDate,lastDate])
        ax[a].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax[a].xaxis.set_major_locator(plt.MultipleLocator(30))
        ax[a].xaxis.set_minor_locator(plt.MultipleLocator(1))
#       if (delta_ts):
#           ax2.append(ax[a].twinx())
#   ax[1].xaxis.set_minor_locator(plt.MultipleLocator(1))

    pn = fit_path+county.replace(' ','_',5)+'.RData'
    if (fit_type == 'ADMB'):
        fit = read_ADMB_rep()
    else:
        fit=pyreadr.read_r(pn)
    print(type(fit))
    diag = fit['diag']
    print(diag)
    meta = fit['meta']
    ests = fit['ests']
    Date0 = get_metadata('Date0',meta)
    Date0 = datetime.strptime(Date0,'%Y-%m-%d')
    pdate = []
    for t in range(0,len(diag.index)):
        pdate.append(mdates.date2num(Date0 + timedelta(days=t)))

    obsI = np.exp(diag['log_obs_cases'])
    preI = np.exp(diag['log_pred_cases'])
    obsD = np.exp(diag['log_obs_deaths'])
    print(diag['log_pred_deaths'])
    preD = np.exp(diag['log_pred_deaths'])

    sigma_logC = np.exp(get_est_or_init('logsigma_logC',ests))
    sigma_logD = np.exp(get_est_or_init('logsigma_logD',ests))

    ax[0].set_title(get_metadata('county',meta))
    ax[0].set_yscale(yscale)
    plot_log_error(ax[0],pdate,diag['log_pred_cases'],sigma_logC)
                  #    ecol=ax[0].get_lines()[line].get_color())
    ax[0].scatter(pdate,obsI)
    ax[0].plot(pdate,preI,color='red')
    tx = prop_scale(ax[0].get_xlim(), 0.05)
    ty = prop_scale(ax[0].get_ylim(), 0.90)
    sigstr = '%s = %.3g'%('$\sigma_I$',sigma_logC)
    ax[0].text(tx,ty,sigstr, ha='left',va='center',fontsize=14)

    ax[1].set_yscale(yscale)
    plot_log_error(ax[1],pdate,diag['log_pred_deaths'],sigma_logD)
    ax[1].scatter(pdate,obsD)
    ax[1].plot(pdate,preD,color='red')
    tx = prop_scale(ax[1].get_xlim(), 0.05)
    ty = prop_scale(ax[1].get_ylim(), 0.90)
    sigstr = '%s = %.3g'%('$\sigma_D$',sigma_logD)
    ax[1].text(tx,ty,sigstr, ha='left',va='center',fontsize=14)

    if (npl > 2):
       sigma_beta = np.exp(get_est_or_init('logsigma_beta',ests))
       plot_error(ax[2],pdate,diag['beta'],sigma_beta)
                 #     ecol=ax[2].get_lines()[line].get_color())
       ax[2].scatter(pdate,diag['beta'])

    if save:
        plt.savefig(fit_path+county.replace(' ','_',5)+'_obsVpred.png',dpi=300)
        plt.show(True)
    else:
        plt.show(True)

def plot_county_dat(dat,County, State,
                  yscale='linear', per_capita=False, delta_ts=True,
                  text_spacer='  ', file = 'prevalence'):
    """ 
    Plots cases and deaths vs time
    """
    mult = 1000
    eps = 1e-5
    pops = pd.read_csv(pop_data_path,header=0)        # get county population

    firstDate = mdates.date2num(FirstNYTDate)
    orderDate = mdates.date2num(CAOrderDate)
    lastDate  = mdates.date2num(EndOfTime)
    print(CAOrderDate,":",orderDate)

    date_list = pd.date_range(start=firstDate,end=lastDate)
    date_lim = [date_list[0],date_list[len(date_list)-1]]

#   dat['Date'] = dat['date'].map(Strptime)

    fig, ax = plt.subplots(2,1,figsize=(6.5,4.5))
    if (per_capita):
        ax[0].set_ylabel('Cases'+' per '+str(mult))
        ax[1].set_ylabel('Deaths'+' per '+str(mult))
    else:
        ax[0].set_ylabel('Cases')
        ax[1].set_ylabel('Deaths')


    ax2 = []
    for a in range(0,len(ax)):
        ax[a].set_xlim([firstDate,lastDate])
        ax[a].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax[a].xaxis.set_major_locator(plt.MultipleLocator(30))
        ax[a].xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax[a].set_yscale(yscale)
        if (delta_ts):
            ax2.append(ax[a].twinx())

    if (delta_ts):
        for a in range(0,len(ax2)):
            ax2[a].set_ylabel("Daily Change")

#   date,county,state,fips,cases,deaths
#   print(dat.columns)
#   print(County,State)
#   cc = Counties['county'][r]
    cc = County
#   st = Counties['state'][r]
#   print('* * county,state:',cc,',',st)
    state_filter = dat['state'].isin([State])
    county_filter = dat['county'].isin([County])
    County_rows = state_filter & county_filter
    cdata = dat[County_rows]

    if (per_capita):
        cp_row = pops['county'] == cc
        pop_size = int(pops[cp_row]['population'])
        cases =  mult*cdata['cases']/pop_size + eps
        deaths =  mult*cdata['deaths']/pop_size + eps
    else :
        cases =  cdata['cases']
        deaths =  cdata['deaths']
    
    Date = []
    for d in cdata['date']:
        Date.append(mdates.date2num(datetime.strptime(d,'%Y-%m-%d').date()))

    
    c = ax[0].plot(Date, cases,label=cc)
    if (delta_ts):
        delta_cases = cases.diff()
        ax2[0].bar(Date,delta_cases,alpha=0.5)
        adc = delta_cases.rolling(window=5).mean()
        ax2[0].plot(Date,adc,linewidth=1)

    #   rescaled second differences
    #   dd_cases = delta_cases.diff()
    #   b = np.nanmin(dd_cases)-np.nanmax(dd_cases)/(np.nanmin(delta_cases)-np.nanmax(delta_cases))
    #   a = np.nanmax(dd_cases)-b*np.nanmax(delta_cases)
    #   dd_cases = a + b*delta_cases 
    #   adc = dd_cases.rolling(window=5).mean()
    #   ax2[0].plot(Date,adc,linewidth=1,color='purple')

    d = ax[1].plot(Date, deaths,label=cc)
    if (delta_ts):
        delta_deaths = deaths.diff()
        ax2[1].bar(Date,delta_deaths,alpha=0.5)
        add = delta_deaths.rolling(window=5).mean()
        ax2[1].plot(Date,add,linewidth=1)

    for a in range(0,len(ax)):
    #   Adjust length of y axis
        ax[a].set_ylim(0,ax[a].get_ylim()[1])
        if (delta_ts):
            ax2[a].set_ylim(0,ax2[a].get_ylim()[1])
    #   Newsome's shelter in place order
        ax[a].plot((orderDate,orderDate),
                   (ax[a].get_ylim()[0], ax[a].get_ylim()[1]),color='black',
                    linewidth=3,alpha=0.5)
    #   ax[a].legend()

    title = 'COVID-19 Prevalence in '+County+' County, '+State
#   fig.text(0.5,1.0,'COVID-19 Prevalence by County',ha='center',va='top')
    fig.text(0.5,1.0,title ,ha='center',va='top')
    fig.text(0.0,0.0,' Data source: New York Times, https://github.com/nytimes/covid-19-data.git.',
             ha='left',va='bottom', fontsize=8)

    mtime = os.path.getmtime("/home/other/nytimes-covid-19-data/us-counties.csv")
    dtime = datetime.fromtimestamp(mtime)
    fig.text(1.0,0.0,'Updated '+str(dtime.date())+' ', ha='right',va='bottom', fontsize=8)

#   signature = 'Graphics by John Sibert'
#   fig.text(1.0,0.025,signature+' ', ha='right',va='bottom', fontsize=8,alpha=0.1)

#   in rcParams: 
#   plt.tight_layout() #pad=1.0, w_pad=1.0, h_pad=5.0)

    save = file
    if save:
        plt.savefig(file+'.png',dpi=300)
    plt.show()

def get_metadata(mdname, meta):
    r = meta['names'].isin([mdname])
    return(meta.data[r].values[0])

def get_estimate(ename, ests):
    r = ests['names'].isin([ename])
    if (r.any() == True):
        return(float(ests.est[r]))
    else:
        return(None)

def get_est_or_init(name,ests):
    v = get_estimate(name,ests) 
    if (isNaN(v)):
        v = get_initpar(n,ests)
        return(v)
    else:
        return(v)

def get_initpar(pname, ests):
    r = ests['names'].isin([pname])
    if (r.any() == True):
        return(float(ests.init[r]))
    else:
        return(None)
   
   
def plot_log_error(ax,x,logy,logsdy,mult=2.0,ecol='0.5'):
    sdyu = np.array(np.exp(logy + mult*logsdy))
    sdyl = np.array(np.exp(logy - mult*logsdy))
    xy = np.array([x,sdyu])
    xy = np.append(xy,np.array([np.flip(x,0),np.flip(sdyl,0)]),axis=1)
    xp = np.transpose(xy).shape
    sd_region = plt.Polygon(np.transpose(xy), alpha=0.5,
                            facecolor='0.9', edgecolor=ecol,lw=1)
    ax.add_patch(sd_region)
   
def plot_error(ax,x,y,sdy,mult=2.0,ecol='0.5'):
    sdyu = np.array(y + mult*sdy)
    sdyl = np.array(y - mult*sdy)
    xy = np.array([x,sdyu])
    xy = np.append(xy,np.array([np.flip(x,0),np.flip(sdyl,0)]),axis=1)
    xp = np.transpose(xy).shape
    sd_region = plt.Polygon(np.transpose(xy), alpha=0.5,
                            facecolor='0.9', edgecolor=ecol,lw=1)
    ax.add_patch(sd_region)
   
def plot_beta_mu(fit_files=['AlamedaCA','SonomaCA'],delta_ts=False,save=True):
    """
    Draws estimated infection and death rate on calander date
    with standard deviation envelopes
    """
    firstDate = FirstNYTDate
    orderDate = CAOrderDate
    lastDate = EndOfTime

    date_list = pd.date_range(start=firstDate,end=lastDate)
    date_lim = [date_list[0],date_list[len(date_list)-1]]

    nfile = len(fit_files)
    if (nfile > 3):
        plt.rcParams['lines.linewidth'] = 2
    fig, ax = plt.subplots(2,1,figsize=(6.5,6.0),sharex=True)
    ax[0].set_ylabel('Infection Rate, 'r'$\beta\ (da^{-1})$')
    ax[1].set_ylabel('Mortality Rate, 'r'$\mu\ (da^{-1})$')
    ax2 = []

    for a in range(0,len(ax)):
        ax[a].set_xlim([firstDate,lastDate])
        ax[a].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax[a].xaxis.set_major_locator(mdates.MonthLocator())
        ax[a].xaxis.set_minor_locator(mdates.DayLocator())
        if (delta_ts):
            ax2.append(ax[a].twinx())
#   ax[1].xaxis.set_minor_locator(plt.MultipleLocator(1))

    max_beta = 0.0
    max_mu = 0.0
    line = -1
    for fn in fit_files:
        pn = fit_path+fn+'.RData'
        print(fn,pn)
        fit=pyreadr.read_r(pn.replace(' ','_',5))
        diag = fit['diag']
        meta = fit['meta']
        est  = fit['ests']
        Date0 = get_metadata('Date0',meta)
        Date0 = datetime.strptime(Date0,'%Y-%m-%d')
        county = get_metadata('county',meta)
        N0 = float(get_metadata('N0',meta))

        ntime = int(get_metadata('ntime',meta))
        tt = list(range(ntime))
        beta = diag['beta']
        max_beta = max(max_beta,beta.max())
        mu = diag['mu']
        max_mu = max(max_mu,mu.max())
        line += 1

        pdate = []
        for t in tt:
            pdate.append(mdates.date2num(Date0 + timedelta(days=t)))
#  produces: [737487.0, 737488.0, 737489.0, ... , 737559.0, 737560.0, 737561.0]
       
#       print(type(pdate),pdate,mdates.date2num(orderDate))
        sipndx = pdate.index(mdates.date2num(orderDate))
#       print(sipndx,pdate[sipndx])

        ax[0].set_ylim(0.0,1.2*max_beta)
        ax[0].plot(pdate,beta,label = short_name(county))
        ax[0].scatter(pdate[sipndx],beta[sipndx],#inestyle='None',
                      marker='|',color='white',edgecolors='face')
        
        mark_ends(ax[0],pdate,beta,fn,'l')
        if (nfile < 3):
            mark_ends(ax[0],pdate,beta,fn,'l')
            sigma_beta = np.exp(get_estimate('logsigma_beta',est))
            plot_error(ax[0],pdate,diag['beta'],sigma_beta,
                       ecol=ax[0].get_lines()[line].get_color())
        if (delta_ts):
            delta_obs_cases = diag['obs_cases'].diff()/N0
            ax2[0].bar(pdate,delta_obs_cases,alpha=0.5,label=fn)
            ax2[0].set_ylabel('New Cases')

        ax[1].set_ylim(0.0,1.2*max_mu)
        ax[1].plot(pdate,mu,label = short_name(county))
        if (nfile < 3):
            mark_ends(ax[1],pdate,mu,fn,'l')
            sigma_mu = np.exp(get_estimate('logsigma_mu',est))
            plot_error(ax[1],pdate,diag['mu'],sigma_mu,
                       ecol=ax[1].get_lines()[line].get_color())
        if (delta_ts):
           delta_obs_deaths = diag['obs_deaths'].diff()
           ax2[1].bar(pdate,delta_obs_deaths,alpha=0.5,label=fn)
           ax2[1].set_ylabel('New Deaths')

#    xmin = min(ax[0].get_xlim()[0],ax[1].get_xlim()[0])
#    xmax = max(ax[0].get_xlim()[1],ax[1].get_xlim()[1])
    for a in range(0,len(ax)):
#       ax[a].set_xlim(xmin,xmax)
    #   Adjust length of y axes
        ax[a].set_ylim(0,ax[a].get_ylim()[1])
        if (delta_ts):
           ax2[a].set_ylim(0,ax2[a].get_ylim()[1])
    #   Newsome's shelter in place order
        ax[a].plot((orderDate,orderDate),
                  (ax[a].get_ylim()[0], ax[a].get_ylim()[1]),color='black',
                   linewidth=3,alpha=0.5)
    #   ax[a].legend(fontsize=10)

    if save:
        plt.savefig(fit_path+'beta_mu'+'.png',format='png',bbox_inches='tight') #dpi=300)
        plt.show(True)
    else:
        plt.show(True)

def QQ_plot(ax,x,y,Q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    xQ = pd.Series(x).quantile(Q)
    yQ = pd.Series(y).quantile(Q)
    ax.scatter(xQ,yQ)
#   xl = ax.get_xlim()
#   yl = ax.get_ylim()
#   ax.plot([xl[0],xl[1]],[yl[0],yl[1]], linewidth=1, color='black')
    qmax = np.max([x,y])
    qmin = np.min([x,y])
    ax.plot([qmin,qmax],[qmin,qmax], linewidth=1, color='red')

def plot_diagnostics(county='AlamedaCA',
                 save = False):
    Q=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    fig, ax = plt.subplots(3,2,figsize=(6.5,9.0))
    county_name = county.replace(' ','_',5)
    pn = fit_path+county_name+'.RData'
    fit=pyreadr.read_r(pn)
    diag = fit['diag']
#   print(diag.columns)
    meta = fit['meta']
#   print(meta)
#   ests  = fit['ests']
#   print(ests)

    obsD = diag['log_obs_deaths']
    preD = diag['log_pred_deaths']

    ax[0,0].scatter(diag['log_obs_cases'].values,
                    diag['log_pred_cases'].values)
    xl = ax[0,0].get_xlim()
    yl = ax[0,0].get_ylim()
    ax[0,0].plot([xl[0],xl[1]],[yl[0],yl[1]], linewidth=1, color='black')
    ax[0,0].set_ylabel(r'$\log\ \widehat{\rm{I}}$')
    ax[0,0].set_xlabel(r'$\log\ \rm{I}$')

    ax[0,1].scatter(preD[obsD>0.0],obsD[obsD>0.0])
    ax[0,1].set_ylabel(r'$\log\ \widehat{\rm{D}}$')
    ax[0,1].set_xlabel(r'$\log\ \rm{D}$')

    qmax = np.max([obsD[obsD>0.0],preD[obsD>0.0]])
    qmin = np.min([obsD[obsD>0.0],preD[obsD>0.0]])
    ax[0,1].plot([qmin,qmax],[qmin,qmax], linewidth=1, color='red')

    QQ_plot(ax[1,0],diag['log_obs_cases'].values,
                    diag['log_pred_cases'].values,Q=Q)
    ax[1,0].set_ylabel(r'$\log\ \widehat{\rm{I}}$ quantiles')
    ax[1,0].set_xlabel(r'$\log\ \rm{I}$ quantiles')

    QQ_plot(ax[1,1], preD[obsD>0.0], obsD[obsD>0.0],Q=Q)
    ax[1,1].set_ylabel(r'$\log\ \widehat{\rm{D}}$ quantiles')
    ax[1,1].set_xlabel(r'$\log\ \rm{D}$ quantiles')

    do_probplot = False
    if (do_probplot): # use stats library
        ax5 = fig.add_subplot(325)
        residuals = diag['log_pred_cases'] - diag['log_obs_cases']
        res5 = stats.probplot(residuals,plot=ax5)
        ax5.set_title('')
        ax5.set_xlabel('Cases Residuals')

        ax6 = fig.add_subplot(326)
        residuals = preD[obsD>0.0] - obsD[obsD>0.0]
        res6 = stats.probplot(residuals,plot=ax6)
        ax6.set_title('')
        ax6.set_xlabel('Deaths Residuals')

    else: # roll your own
        resid = diag['log_pred_cases'] - diag['log_obs_cases']
        qR = resid.quantile(Q)
        normR = pd.Series(stats.norm.ppf(resid))
        qN = normR.quantile(Q)
        ax[2,0].scatter(qR,qN)
        slope, intercept, r, prob, sterrest = stats.linregress(qR,qN)
        ax[2,0].plot(qR,slope*qR+intercept,linewidth=1, color='red')
        ax[2,0].set_xlabel(r'$\log\ \rm{I}$ Residuals')
        ax[2,0].set_ylabel('Normal quantiles')

        resid = preD[obsD>0.0] - obsD[obsD>0.0]
    #   print(resid)
        qR = resid.quantile(Q)
    #   print('qR:',qR)

        normR = pd.Series(stats.norm.ppf(resid))
    #   print('normR:',normR)
        qN = normR.quantile(Q)
    #   print('qN:',qN)

        ax[2,1].scatter(qR,qN)
        slope, intercept, r, prob, sterrest = stats.linregress(qR,qN)
        ax[2,1].plot(qR,slope*qR+intercept, linewidth=1, color='red')
        ax[2,1].set_xlabel(r'$\log\ \rm{D}$ Residuals')
        ax[2,1].set_ylabel('Normal quantiles')

    if save:
        plt.savefig(fit_path+county_name+'_diagnostics'+'.png',dpi=300)
    else:
        plt.show(False)
        plt.show(True)

def isNaN(num):
    return num != num

def pretty_county(s):
    ls = len(s)
    pretty = s[0:(ls-2)]+', '+s[(ls-2):]
    return(pretty.replace('_',' ',5))

def make_fit_table(fit_files=['Alameda','Santa_Clara'],
                 fit_path = '/home/jsibert/Projects/SIR-Models/fits/'):
#   md_cols = ['county','N0','ntime','prop_zero_deaths','fn']
    md_cols = ['county','ntime','prop_zero_deaths','fn','C']
    es_cols = ['logsigma_logP'   , 'logsigma_beta'  , 'logsigma_mu'  ,
               'logsigma_logC','logsigma_logD', 'gamma','mbeta','mmu']
    tt_cols = md_cols + es_cols
    header = ['County','$n$','$p_0$','$f$','$C$',
              '$\sigma_\eta$','$\sigma_\\beta$','$\sigma_\\mu$',
              '$\sigma_I$','$\sigma_D$','$\gamma$','$\\tilde{\\beta}$','$\\tilde{\\mu}$']

    tt = pd.DataFrame(columns=tt_cols,dtype=None)

    func = pd.DataFrame(columns=['fn'])
    gamm = pd.DataFrame(columns=['gamma'])
    mbeta = pd.DataFrame(columns=['mbeta'])
    mmu = pd.DataFrame(columns=['mmu'])
    sigfigs = 3
    for ff in fit_files:
        fn = ff.replace(' ','_',5) 
        pn = fit_path+fn+'.RData'
        fit=pyreadr.read_r(pn)
        ests  = fit['ests']
        meta = fit['meta']
        diag = fit['diag']
        row = pd.Series(index=tt_cols)
        county = get_metadata('county',meta)  
        row['county'] = pretty_county(county)
        for k in range(1,len(tt_cols)):
            n = tt_cols[k]
            v = get_est_or_init(n,ests)
            print(k,n,v)
            if ("logsigma" in n):
                if (v != None):
                    v = float(np.exp(v))
            row.iloc[k] = v

    #   row['N0'] = int(get_metadata('N0',meta))
        row['C'] = get_metadata('convergence',meta)
        row['ntime'] = int(get_metadata('ntime',meta))
        row['prop_zero_deaths'] = round(float(get_metadata('prop_zero_deaths',meta)),sigfigs)
        tt = tt.append(row,ignore_index=True)
        tmp = get_est_or_init('loggamma',ests)
        tmp = np.exp(tmp)
        tmp = float(tmp)
        gamm = np.append(gamm,tmp)
        func = np.append(func,float(get_metadata('fn',meta)))
        beta = diag['beta']
        mbeta = np.append(mbeta,median(beta))
        mu = diag['mu']
        mmu = np.append(mmu,mu.quantile(q=0.5))

    tt['fn'] = func
    tt['gamma'] = gamm
    tt['mbeta'] = mbeta
    tt['mmu'] = mmu

    for c in range(3,len(tt.columns)):
        for r in range(0,tt.shape[0]):
           if (tt.iloc[r,c] != None):
               tt.iloc[r,c] = round(float(tt.iloc[r,c]),sigfigs)
#   tt = tt.sort_values(by='N0',ascending=False)#,inplace=True)

    row = pd.Series(None,index=tt.columns)
    row['county'] = 'Median'
    for n in tt.columns:
        if (n != 'county'):
            mn = tt[n].quantile()
            row[n] = mn
    tt = tt.append(row,ignore_index=True)
    print(tt)

    tex = fit_path+'fit_table.tex'
    ff = open(tex, 'w')
    ff.write(tabulate(tt, header, tablefmt="latex_raw",showindex=False))
#   tt.to_latex(buf=tex,index=False,index_names=False,longtable=False,
#               header=header,escape=False,#float_format='{:0.4f}'.format
#               na_rep='',column_format='lrrrrrrrrrrr')
    print('Fit table written to file',tex)

##############################################################################
##############################################################################
if __name__ == '__main__':


#   for c in LargestCACounties:
#       make_ADMB_dat(county_dat,c)

#   plot_county_dat(county_dat,
#                 Counties=county_state.iloc[[1,7]],
#                 death_threshold=1, cases_threshold=1,file='county_plot',
#                 delta_ts=True,per_capita=False)

#   plot_counties(county_dat,Counties=['Contra Costa'],
#                 death_threshold=1, cases_threshold=10,file='county_plot',delta_ts=True)
#   plot_counties(county_dat,Counties=LargestCACounties,
#                 death_threshold=1, cases_threshold=10,
#                 per_capita=False,yscale='log',file='BigCA')
#   plot_counties(county_dat,Counties=BayAreaCounties,
#                 death_threshold=1, cases_threshold=10, file='BayArea')
#   plot_counties(county_dat,Counties=NewYorkCounties,
#                 death_threshold=5, cases_threshold=50)
#   plot_counties(county_dat,Counties=['Orange'],
#                 death_threshold=1, cases_threshold=10,file='Orange')

#   make_fit_table(["New_York_City","Los_Angeles", "San_Bernardino", "Alameda",
#                 "Sacramento","Contra_Costa", "Ventura","San_Mateo","San_Joaquin",
#                 "Stanislaus","Sonoma","Marin"])

#   pop = get_county_pop('Alameda','California')

#   update_dat()
#   make_dat_file()

#   plot_beta_mu(largest_us_counties,save=True)
#   plot_beta_mu(['MaricopaAZ','WayneMI'])

#   make_fit_table(largest_us_counties,
#                fit_path = '/home/jsibert/Projects/SIR-Models/fits/')

#   plot_diagnostics(save=True)
#   for c in largest_us_counties:
#       plot_diagnostics(c,save=True)
#   for c in big_county_list:
#       plot_county_fit(c,yscale='linear',save=True)
#   plot_county_fit('Miami-DadeFL',yscale='log',save=True)

#   for c in largest_us_counties:
#       check_delta_t(county_dat,c)
#   plot_county_dat(county_dat,
#                   County='Cook',State='Illinois',file='CookIL')
#                   County='Broward',State='Florida',file='BrowardFL')
#                   County='Miami-Dade',State='Florida',file='MiamiDadeFL')
#                   County='Harris',State='Texas',file='HarrisTX')
#                   County='Alameda',State='California',file='AlamedaCA')
                 
#   plot_beta_mu(['CookIL'],plot_mu=False, delta_ts=True,save=True)
    plot_county_fit('AlamedaCA',yscale='linear',save=False,fit_type='ADMB')
#   plot_diagnostics('CookIL',save=False)

#   fit = read_ADMB_rep()
#   print(fit['diag'])


else:
   print('type something')

