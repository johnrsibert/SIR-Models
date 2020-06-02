#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jsibert
"""
import pyreadr
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc
import numpy as np
import os
from io import StringIO
import scipy.stats as stats
from sigfig import round
#from tabulate import tabulate

plt.style.use('file:///home/jsibert/.config/matplotlib/john.mplstyle')

LargestCACounties = ['Los Angeles', 'San Diego', 'Orange', 'Riverside',
                         'San Bernardino', 'Santa Clara', 'Alameda',
                         'Sacramento', 'Contra Costa']
NewYorkCounties = ['New York City']#,'Tompkins']
EastBayCounties = ['Santa Clara','Alameda','Contra Costa','Sonoma','Napa']
BayAreaCounties = EastBayCounties + ['San Mateo','San Francisco','Marin']

big_county_list = ["New York City","Los Angeles","San Diego",
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
EndOfTime = datetime.strptime('2020-05-31','%Y-%m-%d')

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
    population = int(dat[County_rows]['population'])
    return(population)


def check_delta_t(dat,County):
    State = 'California'
    state_filter = dat['state'].isin([State])
    county_filter = dat['county'].isin([County])
    County_rows = state_filter & county_filter
    cdat = dat[County_rows]

    Date = cdat['date'].map(Strptime)
    dDate = Date.diff()
    n = len(dDate) - 1
    print('Maximum delta t for',County,State,':',max(dDate[-n:]))
    return(max(dDate[-n:]).days)

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

def make_dat_file (County='Los Angeles',State='California',ST='CA',cpath='../county-state.csv'):
    """ Generate input data for analysis by ADMB or TMB models
        dat: pandas object read by another function
        County: string containing the name of the county in California
    """

#   maxdt = check_delta_t(dat,County)
#   if (maxdt > 1):
#       print('WARNING: timstep greater than one for',County,'County')

    
#   pops = pd.read_csv(pop_data_path,header=0)        # get county population
#   cp_row = pops['county'] == County 
    
    dat = pd.read_csv(cpath,header=0)
    
    print(dat.columns)
#   print(dat)
#   print(dat["ST"])
    population = get_county_pop(County, State)
    print(population)

    mtime = os.path.getmtime("/home/other/nytimes-covid-19-data/us-counties.csv")
    dtime = datetime.fromtimestamp(mtime)
    update_stamp = str(dtime.date())
 

#   print(dat.head())
#   get county cases & deaths    
    state_filter = dat['ST'].isin([ST])
    county_filter = dat['county'].isin([County])
    County_rows = state_filter & county_filter
    print(County_rows)
    dfile = County.replace(' ','_',5)+ST+'.dat'
    print(County, ST,dfile)

    O = open(dfile,'w')
    O.write('# county\n %s\n'%County.replace(' ','_',5))
    O.write('# updateed from https://github.com/nytimes/covid-19-data.git\n')
    O.write(' %s\n'%update_stamp)
    O.write('# population (N0)\n %10d\n'%population)
    print(dat[County_rows])
#   r0 = dat['date'][County_rows].index[0]
    return(0)
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

def plot_county_fit(county,
    #             death_threshold=1, cases_threshold=1,
                  yscale='log', per_capita=False, delta_ts=False,
                  text_spacer='  ', file = None):
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
    fig, ax = plt.subplots(2,1,figsize=(6.5,6.0))
    if (per_capita):
        ax[0].set_ylabel('Cases'+' per '+str(mult))
        ax[1].set_ylabel('Deaths'+' per '+str(mult))
    else:
        ax[0].set_ylabel('Cases')
        ax[1].set_ylabel('Deaths')


    for a in range(0,len(ax)):
        ax[a].set_xlim([firstDate,lastDate])
        ax[a].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax[a].xaxis.set_major_locator(plt.MultipleLocator(30))
        ax[a].xaxis.set_minor_locator(plt.MultipleLocator(1))
#       if (delta_ts):
#           ax2.append(ax[a].twinx())
#   ax[1].xaxis.set_minor_locator(plt.MultipleLocator(1))


#   ax[0].set_xlabel('Time (days after n>%.0f)'%cases_threshold)
#   ax[1].set_xlabel('Time (days after n>%.0f)'%death_threshold)

    pn = fit_path+county.replace(' ','_',5)+'.RData'
    fit=pyreadr.read_r(pn)
    diag = fit['diag']
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
    preD = np.exp(diag['log_pred_deaths'])

    sigma_logC = get_est_or_init('sigma_logC',ests)
    sigma_logD = get_est_or_init('sigma_logD',ests)

    ax[0].set_title(get_metadata('county',meta))
    ax[0].set_yscale(yscale)
    plot_log_error(ax[0],pdate,diag['log_pred_cases'],sigma_logC)
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

    plt.show()

def plot_county_dat(dat,Counties,
                  death_threshold=1, cases_threshold=1,
                  yscale='linear', per_capita=False, delta_ts=True,
                  text_spacer='  ', file = None):
    """ 
    Plots cases and deaths vs time from threshold by countiy
    Counties: list of California counties to plotted
    """

#   get county population       
#   pops = pd.read_csv(pop_data_path,header=0)
#   pops = get_county_pop(County, state='California')
    mult = 1000
    eps = 1e-5

#   firstDate = FirstNYTDate
#   orderDate = CAOrderDate
#   lastDate = EndOfTime

    firstDate = mdates.date2num(FirstNYTDate)
    orderDate = mdates.date2num(CAOrderDate)
    lastDate  = mdates.date2num(EndOfTime)
    print(CAOrderDate,":",orderDate)

    date_list = pd.date_range(start=firstDate,end=lastDate)
    date_lim = [date_list[0],date_list[len(date_list)-1]]

#   dat['Date'] = dat['date'].map(Strptime)

    fig, ax = plt.subplots(2,1,figsize=(6.5,6.0))
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
    print(dat.columns)
    print('Counties:',len(Counties))
    print(Counties)
    print(Counties.index)
    for r in Counties.index:
        cc = Counties['county'][r]
        st = Counties['state'][r]
        print('* * county,state:',cc,',',st)
        state_filter = dat['state'].isin([st])
        county_filter = dat['county'].isin([cc])
    #   threshold_filter = dat[Column] > threshold
        County_rows = state_filter & county_filter# & threshold_filter
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
        #   adc =moving_average(delta_cases, n=7)
        #   print(adc)
        #   ax2[0].plot(Date,adc)

        d = ax[1].plot(Date, deaths,label=cc)
        if (delta_ts):
            delta_deaths = deaths.diff()
            ax2[1].bar(Date,delta_deaths,alpha=0.5)

 
    for a in range(0,len(ax)):
    #   Adjust length of y axis
        ax[a].set_ylim(0,ax[a].get_ylim()[1])
        if (delta_ts):
            ax2[a].set_ylim(0,ax2[a].get_ylim()[1])
    #   Newsome's shelter in place order
        ax[a].plot((orderDate,orderDate),
                   (ax[a].get_ylim()[0], ax[a].get_ylim()[1]),color='black',
                    linewidth=3,alpha=0.5)
        ax[a].legend()

    mtime = os.path.getmtime("/home/other/nytimes-covid-19-data/us-counties.csv")
    dtime = datetime.fromtimestamp(mtime)
    fig.text(0.5,1.0,'COVID-19 Prevalence by County',ha='center',va='top')
    fig.text(0.0,0.0,' Data source: New York Times, https://github.com/nytimes/covid-19-data.git.',
             ha='left',va='bottom', fontsize=8)
    fig.text(1.0,0.0,'Updated '+str(dtime.date())+' ', ha='right',va='bottom', fontsize=8)

#   in rcParams
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
   
   
def plot_log_error(ax,x,logy,logsdy,mult=2.0):
    sdyu = np.array(np.exp(logy + mult*logsdy))
    sdyl = np.array(np.exp(logy - mult*logsdy))
    xy = np.array([x,sdyu])
    xy = np.append(xy,np.array([np.flip(x,0),np.flip(sdyl,0)]),axis=1)
    xp = np.transpose(xy).shape
    sd_region = plt.Polygon(np.transpose(xy), alpha=0.5,
                            facecolor='0.9', edgecolor='0.5')
    ax.add_patch(sd_region)
   
def plot_error(ax,x,y,sdy,mult=2.0,ecol='0.5'):
    sdyu = np.array(y + mult*sdy)
    sdyl = np.array(y - mult*sdy)
    xy = np.array([x,sdyu])
    xy = np.append(xy,np.array([np.flip(x,0),np.flip(sdyl,0)]),axis=1)
    xp = np.transpose(xy).shape
    sd_region = plt.Polygon(np.transpose(xy), alpha=0.5,
                            facecolor='0.9', edgecolor=ecol)
    ax.add_patch(sd_region)
   
def plot_beta_mu(fit_files=['Alameda'],delta_ts=False,save=True):
    """
    Draws estimated infection and death rate on calander date
    with standard deviation envelopes
    """
    firstDate = FirstNYTDate
    orderDate = CAOrderDate
    lastDate = EndOfTime

    date_list = pd.date_range(start=firstDate,end=lastDate)
    date_lim = [date_list[0],date_list[len(date_list)-1]]
#                              sharex=True
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
        fit=pyreadr.read_r(pn.replace(' ','_',5))
        diag = fit['diag']
        meta = fit['meta']
        est  = fit['ests']
        Date0 = get_metadata('Date0',meta)
        Date0 = datetime.strptime(Date0,'%Y-%m-%d')
        county = get_metadata('county',meta)

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
       
#        print(type(pdate),pdate,mdates.date2num(orderDate))
        sipndx = pdate.index(mdates.date2num(orderDate))
#        print(sipndx,pdate[sipndx])

        ax[0].set_ylim(0.0,1.2*max_beta)
        ax[0].plot(pdate,beta,label = short_name(county))
        ax[0].scatter(pdate[sipndx],beta[sipndx],#inestyle='None',
                      marker='|',color='white',edgecolors='face')
        
        mark_ends(ax[0],pdate,beta,fn,'l')
        sigma_beta = get_estimate('sigma_beta',est)
        plot_error(ax[0],pdate,diag['beta'],sigma_beta,
                   ecol=ax[0].get_lines()[line].get_color())
        if (delta_ts):
            delta_obs_cases = diag['obs_cases'].diff()
            ax2[0].bar(pdate,delta_obs_cases,alpha=0.5,label=fn)
            ax2[0].set_ylabel('New Cases')


        ax[1].set_ylim(0.0,0.25*max_mu)
        ax[1].plot(pdate,mu,label = short_name(county))
        mark_ends(ax[1],pdate,mu,fn,'r')
        sigma_mu = get_estimate('sigma_mu',est)
        plot_error(ax[1],pdate,diag['mu'],sigma_mu,
                   ecol=ax[1].get_lines()[line].get_color())
        if (delta_ts):
            delta_obs_deaths = diag['obs_deaths'].diff()
            ax2[1].bar(pdate,delta_obs_deaths,alpha=0.5,label=fn)
            ax2[1].set_ylabel('New Deaths')

#    xmin = min(ax[0].get_xlim()[0],ax[1].get_xlim()[0])
#    xmax = max(ax[0].get_xlim()[1],ax[1].get_xlim()[1])
    for a in range(1,len(ax)):
#        ax[a].set_xlim(xmin,xmax)
    #   Adjust length of y axes
        ax[a].set_ylim(0,ax[a].get_ylim()[1])
        if (delta_ts):
           ax2[a].set_ylim(0,ax2[a].get_ylim()[1])
    #   Newsome's shelter in place order
        ax[a].plot((orderDate,orderDate),
                  (ax[a].get_ylim()[0], ax[a].get_ylim()[1]),color='black',
                   linewidth=3,alpha=0.5)
    #    ax[a].legend(fontsize=10)

    if save:
        plt.savefig(fit_path+'beta_mu'+'.pdf',format='pdf',bbox_inches='tight') #dpi=300)
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
    ax.plot([qmin,qmax],[qmin,qmax], linewidth=1, color='black')

def plot_diagnostics(county='Alameda',
                 save = False):
    fig, ax = plt.subplots(3,2,figsize=(6.5,9.0))
    county_name = county.replace(' ','_',5)
    pn = fit_path+county_name+'.RData'
    fit=pyreadr.read_r(pn)
    diag = fit['diag']
    print(diag.columns)
    meta = fit['meta']
    print(meta)
    ests  = fit['ests']
    print(ests)

    obsD = diag['log_obs_deaths']
    preD = diag['log_pred_deaths']

    do_probplot = True
    if (do_probplot):
        ax5 = fig.add_subplot(325)
        residuals = diag['log_pred_cases'] - diag['log_obs_cases']
        res5 = stats.probplot(residuals,plot=ax5)
#   print('res5:',res5)
        ax5.set_title('')
        ax5.set_xlabel('Cases Residuals')

        ax6 = fig.add_subplot(326)
        residuals = preD[obsD>0.0] - obsD[obsD>0.0]
        res6 = stats.probplot(residuals,plot=ax6)
    #   print('res6:',res6)
        ax6.set_title('')
        ax6.set_xlabel('Deaths Residuals')

    else:
        ax5 = fig.add_subplot(325)
        ax5.plot(diag['obs_cases'])
        ax6 = fig.add_subplot(326)

    ax[1,0].scatter(diag['log_obs_cases'].values,
                    diag['log_pred_cases'].values)
    xl = ax[1,0].get_xlim()
    yl = ax[1,0].get_ylim()
    ax[1,0].plot([xl[0],xl[1]],[yl[0],yl[1]], linewidth=1, color='black')
#   qmax = np.max(diag['log_obs_cases'], diag['log_pred_cases'])
#   qmin = np.min(diag['log_obs_cases'].values, diag['log_pred_cases'].values)
#   ax[1,0].plot([qmin,qmax],[qmin,qmax], linewidth=1, color='black')
    ax[1,0].set_xlabel('log Observed Cases')
    ax[1,0].set_ylabel('log predicted Cases')
    ax[1,0].set_ylabel(r'$\log\ \widehat{\rm{I}}$')
    ax[1,0].set_xlabel(r'$\log\ \rm{I}$')

    ax[1,1].scatter(preD[obsD>0.0],obsD[obsD>0.0])
    ax[1,1].set_ylabel(r'$\log\ \widehat{\rm{D}}$')
    ax[1,1].set_xlabel(r'$\log\ \rm{D}$')

    qmax = np.max([obsD[obsD>0.0],preD[obsD>0.0]])
    qmin = np.min([obsD[obsD>0.0],preD[obsD>0.0]])
    ax[1,1].plot([qmin,qmax],[qmin,qmax], linewidth=1, color='black')

    QQ_plot(ax[0,0],diag['log_obs_cases'].values,
                    diag['log_pred_cases'].values)
    ax[0,0].set_ylabel(r'$\log\ \widehat{\rm{I}}$ quantiles')
    ax[0,0].set_xlabel(r'$\log\ \rm{I}$ quantiles')

    QQ_plot(ax[0,1], preD[obsD>0.0], obsD[obsD>0.0])
    ax[0,1].set_ylabel(r'$\log\ \widehat{\rm{D}}$ quantiles')
    ax[0,1].set_xlabel(r'$\log\ \rm{D}$ quantiles')

    if save:
        plt.savefig(fit_path+county_name+'_diagnostics'+'.png',dpi=300)
        plt.show(False)
    else:
        plt.show(True)

def isNaN(num):
    return num != num

def make_fit_table(fit_files=['Alameda','Santa_Clara'],
                 fit_path = '/home/jsibert/Projects/SIR-Models/fits/'):
    tt_cols = ['county','N0','ntime','prop_zero_deaths','fn'  ,
               'sigma_logP'   , 'sigma_beta'  ,'sigma_mu',
               'sigma_logC','sigma_logD', 'gamma']
    header = ['County','$N_0$','$n$','$p_0$','$f$',
              '$\sigma_\eta$','$\sigma_\\beta$','$\sigma_\mu$',
              '$\sigma_I$','$\sigma_D$','$\gamma$','$\\tilde{\\beta}$','$\\tilde{\mu}$']

    tt = pd.DataFrame(columns=tt_cols,dtype=None)

    func = pd.DataFrame(columns=['fn'])
    gamm = pd.DataFrame(columns=['gamma'])
    mbeta = pd.DataFrame(columns=['mbeta'])
    mmu = pd.DataFrame(columns=['mmu'])
#   lgndx = tt_cols.index('loggamma')
    sigfigs = 3
    for ff in fit_files:
        fn = ff.replace(' ','_',5) 
        pn = fit_path+fn+'.RData'
        fit=pyreadr.read_r(pn)
        ests  = fit['ests']
        meta = fit['meta']
        diag = fit['diag']
        row = [None]*len(tt_cols)
        county = get_metadata('county',meta)  
        row[0] = county.replace('_',' ',5) 
        for k in range(1,len(tt_cols)):
            n = tt_cols[k]
            v = get_est_or_init(n,ests)
            row[k] = round(v,sigfigs)
        row[1] = int(get_metadata('N0',meta))
        row[2] = int(get_metadata('ntime',meta))
        row[3] = round(float(get_metadata('prop_zero_deaths',meta)),sigfigs)
        tt = np.append(tt,np.array([row]),axis=0)
        tmp = get_est_or_init('loggamma',ests)
        tmp = np.exp(tmp)
        tmp = round(float(tmp),sigfigs)
        gamm = np.append(gamm,tmp)
#       gamm = np.append(gamm,round(np.exp(float(get_est_or_init('loggamma',ests))),sigfigs))
        func = np.append(func,round(       get_metadata('fn',meta),sigfigs))
        beta = diag['beta']
        mbeta = np.append(mbeta,round(float(beta.quantile(q=0.5)),sigfigs))
        mu = diag['mu']
        mmu = np.append(mmu,round(float(mu.quantile(q=0.5)),sigfigs))

    tt = pd.DataFrame(tt,columns=tt_cols)#,dtype=float)
    tt['fn'] = func
    tt['gamma'] = gamm
    tt['mbeta'] = mbeta
    tt['mmu'] = mmu

    tex = fit_path+'fit_table.tex'
    ff = open(tex, 'w')
    ff.write(tabulate(tt, header, tablefmt="latex_raw",showindex=False))
#   tt.to_latex(buf=tex,index=False,index_names=False,longtable=False) #,header=header)
    print('Fit table written to file',tex)

##############################################################################
##############################################################################
if __name__ == '__main__':

    make_dat_file()

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
#   plot_beta_mu(['Los Angeles']) #,'San_Mateo' ])#,delta_ts=True)
#   plot_beta_mu([ 'Contra_Costa', ] ,delta_ts=True)
#   plot_beta_mu([ 'Alameda', 'Contra_Costa', 'Los Angeles', 'Marin',
#                  'New York City', 'Sacramento', 'San Bernardino',
#                  'San Joaquin', 'San_Mateo', 'Sonoma', 'Stanislaus',
#                  'Ventura'],

#   make_fit_table(["New_York_City","Los_Angeles", "San_Bernardino", "Alameda",
#                 "Sacramento","Contra_Costa", "Ventura","San_Mateo","San_Joaquin",
#                 "Stanislaus","Sonoma","Marin"])

#   pop = get_county_pop('Alameda','California')
#   plot_beta_mu(big_county_list,
#                delta_ts=False,save=True)

#   make_fit_table(big_county_list)

#   make_fit_table(county_state['county'])

#   for c in ['San Bernardino','San Diego','San_Francisco']:
#       plot_diagnostics(c,save=True)
#   plot_diagnostics('New_York_City')

#   plot_county_fit('Los Angeles',yscale='linear')

#   for c in LargestCACounties:
#       check_delta_t(county_dat,c)
#   for c in range(0,len(county_state)):
#       make_ADMB_dat(county_dat,county_state.loc[c])

else:
   print('type something')

