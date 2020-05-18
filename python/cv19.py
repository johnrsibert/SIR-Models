#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:03:18 2020

@author: jsibert

Plots corona virus cases and deaths by county

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

plt.style.use('file:///home/jsibert/.config/matplotlib/john.mplstyle')

"""
states_path = "us-states.csv"
print('processing states file',states_path)
states = pd.read_csv(states_path,header=0)
print('states:',states.shape)
"""
pop_data_path = '../CA-populations.csv'
LargestCACounties = ['Los Angeles', 'San Diego', 'Orange', 'Riverside',
                         'San Bernardino', 'Santa Clara', 'Alameda',
                         'Sacramento', 'Contra Costa']
NewYorkCounties = ['New York City']#,'Tompkins']
EastBayCounties = ['Santa Clara','Alameda','Contra Costa','Sonoma','Napa']
BayAreaCounties = EastBayCounties + ['San Mateo','San Francisco','Marin']
counties_path = "../us-counties.csv"
#print('processing county population file',counties_path)
county_dat = pd.read_csv(counties_path,header=0)

def make_ADMB_dat(dat,County):
    """ Generate input data for analysis by ADMB or TMB models
        dat: pandas object read by another function
        County: string containing the name of the county in California
    """
    State = 'California'
    pops = pd.read_csv(pop_data_path,header=0)        # get county population
    cp_row = pops['county'] == County 
    population = int(pops[cp_row]['population'])

    mtime = os.path.getmtime("/home/other/nytimes-covid-19-data/us-counties.csv")
    dtime = datetime.fromtimestamp(mtime)
    update_stamp = str(dtime.date())
 
#   O = open('ad.dat','w')
    O = open(County.replace(' ','_')+'.dat','w')
    O.write('# county\n %s\n'%County.replace(' ','_'))
    O.write('# updateed from https://github.com/nytimes/covid-19-data.git\n')
    O.write(' %s\n'%update_stamp)
    O.write('# population (N0)\n %10d\n'%population)

    print(dat.head())
#   get county cases & deaths    
    state_filter = dat['state'].isin([State])
    county_filter = dat['county'].isin([County])
    County_rows = state_filter & county_filter

    r0 = dat['date'][County_rows].index[0]
    date0 = dat['date'][r0]
    print('date0',date0)
    O.write('# date zero\n %s\n'%date0)
    print(dat[County_rows])
    ntime = len(dat[County_rows])-1
    print('----------------ntime',ntime)
    O.write('# ntime\n %4d\n'%ntime)
    print(dat[County_rows].index)
    print('date','index','cases','deaths')
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


def plot_counties(dat,Counties,
                  death_threshold=1, cases_threshold=1,
                  yscale='log', per_capita=False, delta_ts=False,
                  text_spacer='  ', file = None):
    """ Plots cases and deaths vs time from threshold by countiy
        Counties: list of California counties to plotted
    """

    def Strptime(x):
        s = str(x)
        y = datetime.strptime(x,'%Y-%m-%d')
        return(y)
        
    pops = pd.read_csv(pop_data_path,header=0)        # get county population
    mult = 1000
    eps = 1e-5

    dat['Date'] = dat['date'].map(Strptime)

    fig, ax = plt.subplots(2,1,figsize=(6.5,6.0))
    if (per_capita):
        ax[0].set_ylabel('Cases'+' per '+str(mult))
        ax[1].set_ylabel('Deaths'+' per '+str(mult))
    else:
        ax[0].set_ylabel('Cases')
        ax[1].set_ylabel('Deaths')

    ax[0].set_xlabel('Time (days after n>%.0f)'%cases_threshold)
    ax[1].set_xlabel('Time (days after n>%.0f)'%death_threshold)

    ax2 = []
    for a in range(0,len(ax)):
        if (delta_ts):
            ax2.append(ax[a].twinx())

    for County in Counties:
        deaths = make_t0_series(dat,County,'deaths',death_threshold)
        cases  = make_t0_series(dat,County,'cases',cases_threshold)
        if (per_capita):
            cp_row = pops['county'] == County
            pop_size = int(pops[cp_row]['population'])
            deaths['deaths'] = mult*deaths['deaths']/pop_size + eps
            cases['cases'] = mult*cases['cases']/pop_size + eps
        
        ax[0].set_yscale(yscale)
        c = ax[0].plot(cases['time'], cases['cases'])
        clen = len(cases['cases'])-1
        ax[0].text(clen,cases['cases'].iloc[clen],text_spacer+County,
                   va='center',color=c[0].get_color())
        if (delta_ts):
            delta_cases = cases['cases'].diff()
            ax2[0].bar(range(len(delta_cases)),delta_cases,alpha=0.75)

        ax[1].set_yscale(yscale)
        d = ax[1].plot(deaths['time'], deaths['deaths'])
        dlen = len(deaths['deaths'])-1
        ax[1].text(dlen,deaths['deaths'].iloc[dlen],text_spacer+County,
                   va='center',color=d[0].get_color())
        if (delta_ts):
            delta_deaths = deaths['deaths'].diff()
            ax2[1].bar(range(len(delta_deaths)),delta_deaths,alpha=0.75)

    if (yscale == 'log' and not per_capita):
        dtimes = [1.0,2.0,4.0,8.0,16.0] # doubling time
        ln2 = np.log(2.0)
    
        rates = []  # exponential rates equivalent to doubling time
        for t in range(0,len(dtimes)):
            rates.append(ln2/dtimes[t])

        for p in range(len(ax)):
            tlim = list(ax[p].get_xlim())
            cylim = ax[p].get_ylim()
            dylim = 0.9*cylim[1]
            if (p == 0):
                threshold = cases_threshold
            else:
                threshold = death_threshold

            slopes = []
            for r in range(0,len(rates)):
                slopes.append(threshold*np.exp(rates[r]*tlim[1]))
                ax[p].plot([0,tlim[1]] , [threshold,slopes[r]], color='0.5',lw=1)
                if (slopes[r] < dylim):
                    ax[p].text(tlim[1] , slopes[r],text_spacer+'%.0fd'%dtimes[r], 
                               va='center',
                               color='0.5',fontsize=10)
                else:
                    tt = (np.log(dylim)-np.log(threshold))/rates[r]
                    ax[p].text(tt , dylim,text_spacer+'%.0fd'%dtimes[r], va='bottom',
                               color='0.5',fontsize=10)
            ax[p].set_ylim(cylim)
 
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
    return( meta.data[r].values[0])

def get_estimate(ename, ests):
    r = ests['names'].isin([ename])
    if (r.any() == True):
        return(float(ests.est[r]))
    else:
        return(None)
   
def plot_error(ax,x,logy,logsdy,mult=2.0):
    sdyu = np.array(np.exp(logy + mult*logsdy))
    sdyl = np.array(np.exp(logy - mult*logsdy))
    xy = np.array([x,sdyu])
    xy = np.append(xy,np.array([np.flip(x,0),np.flip(sdyl,0)]),axis=1)
    xp = np.transpose(xy).shape
    sd_region = plt.Polygon(np.transpose(xy), facecolor='0.9', edgecolor='0.5')
    ax.add_patch(sd_region)

def plot_beta_mu(fit_files=['Alameda'],delta_ts=False,
                 fit_path = '/home/jsibert/Projects/SIR-Models/fits/'):
    """
    Draws estimated infection and death rate on calander date
    with standard deviation envelopes
    """
    firstDate = datetime.strptime('2020-01-21','%Y-%m-%d')
    lastDate= datetime.strptime('2020-05-31','%Y-%m-%d')
    orderDate= datetime.strptime('2020-03-19','%Y-%m-%d')


    date_list = pd.date_range(start=firstDate,end=lastDate)
    date_lim = [date_list[0],date_list[len(date_list)-1]]
#                              sharex=True
    fig, ax = plt.subplots(2,1,figsize=(6.5,6.0),sharex=True)
    ax[0].set_ylabel('Infecrion Rate, 'r'$\beta\ (da^{-1})$')
    ax[1].set_ylabel('Mortality Rate, 'r'$\mu\ (da^{-1})$')
    ax2 = []

    for a in range(0,len(ax)):
        ax[a].set_xlim([firstDate,lastDate])
        ax[a].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax[a].xaxis.set_major_locator(plt.MultipleLocator(30))
        ax[a].xaxis.set_minor_locator(plt.MultipleLocator(1))
        if (delta_ts):
            ax2.append(ax[a].twinx())
#   ax[1].xaxis.set_minor_locator(plt.MultipleLocator(1))

    for fn in fit_files:
        pn = fit_path+fn+'.RData'
        fit=pyreadr.read_r(pn)
        diag = fit['diag']
        meta = fit['meta']
        est  = fit['est']
        Date0 = get_metadata('Date0',meta)
        Date0 = datetime.strptime(Date0,'%Y-%m-%d')

        ntime = int(get_metadata('ntime',meta))
        tt = list(range(ntime))
        beta = np.exp(diag['logbeta'])
        mu = np.exp(diag['logmu'])

        pdate = []
        for t in tt:
            pdate.append(mdates.date2num(Date0 + timedelta(days=t)))

        ax[0].plot(pdate,beta,label = fn)
        sigma_logbeta = get_estimate('sigma_logbeta',est)
        plot_error(ax[0],pdate,diag['logbeta'],sigma_logbeta)
        if (delta_ts):
            delta_obs_cases = diag['obs_cases'].diff()
            ax2[0].scatter(pdate,delta_obs_cases,alpha=0.5,label=fn)

        ax[1].plot(pdate,mu,label = fn)
        sigma_logmu = get_estimate('sigma_logmu',est)
        plot_error(ax[1],pdate,diag['logmu'],sigma_logmu)
        if (delta_ts):
            delta_obs_deaths = diag['obs_deaths'].diff()
            ax2[1].scatter(pdate,delta_obs_deaths,alpha=0.5,label=fn)

    for a in range(0,len(ax)):
    #   Adjust length of y axis
        ax[a].set_ylim(0,ax[a].get_ylim()[1])
    #   Newsome's shelter in place order
        ax[a].plot((orderDate,orderDate),
                   (ax[a].get_ylim()[0], ax[a].get_ylim()[1]),color='black')
        ax[a].legend()
    plt.show()

def plot_diagnostics(county='Alameda',
                 fit_path = '/home/jsibert/Projects/SIR-Models/fits/',
                 save = True):
    fig, ax = plt.subplots(3,2,figsize=(6.5,9.0))
    county_name = county.replace(' ','_')
    pn = fit_path+county_name+'.RData'
    fit=pyreadr.read_r(pn)
    diag = fit['diag']
#   print(diag.columns)
#   meta = fit['meta']
#   est  = fit['est']

    obsD = diag['log_obs_deaths']
    preD = diag['log_pred_deaths']

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

    ax[1,0].plot([np.min(diag['log_obs_cases']),np.max(diag['log_obs_cases'])],
                 [np.min(diag['log_obs_cases']),np.max(diag['log_obs_cases'])],
                 linewidth=1, color='black')
    ax[1,0].scatter(diag['log_obs_cases'].values,
                    diag['log_pred_cases'].values)
    ax[1,0].set_xlabel('log Observed Cases')
    ax[1,0].set_ylabel('log predicted Cases')

#   ax[1,1].plot([np.min(diag['log_obs_deaths'].values),np.max(diag['log_obs_deaths'])],
#                [np.min(diag['log_obs_deaths'].values),np.max(diag['log_obs_deaths'])],
#                linewidth=1, color='black')
#   ax[1,1].scatter(diag['log_obs_deaths'].values,
#                   diag['log_pred_deaths'].values)
    ax[1,1].plot([np.min(preD[obsD>0.0]),np.max(obsD[obsD>0.0])],
                 [np.min(preD[obsD>0.0]),np.max(obsD[obsD>0.0])],
                 linewidth=1, color='black')
    ax[1,1].scatter(preD[obsD>0.0],obsD[obsD>0.0])
    ax[1,1].set_xlabel('log Observed Deaths')
    ax[1,1].set_ylabel('log predicted Deaths')

    if save:
        plt.savefig(fit_path+county_name+'_diagnostics'+'.png',dpi=300)
        plt.show(False)
    else:
        plt.show()

def make_fit_table(fit_files=['Alameda','Santa_Clara'],
                 fit_path = '/home/jsibert/Projects/SIR-Models/fits/'):
#   tt_cols = ['county','sigma_logP','sigma_logbeta','sigma_logmu','loggamma']
    tt_cols = ['county','fn','sigma_logP','sigma_logbeta','sigma_logmu','loggamma','gamma']
    tt = pd.DataFrame(columns=tt_cols,dtype=None)

    func = pd.DataFrame(columns=['fn'])
    gamm = pd.DataFrame(columns=['gamma'])
    sigfigs = 3
    for fn in fit_files:
        pn = fit_path+fn+'.RData'
        fit=pyreadr.read_r(pn)
        est  = fit['est']
        meta = fit['meta']
        row = [None]*len(tt_cols)
        row[0] = fn
        for n in range(1,len(tt_cols)):
            v = get_estimate(tt_cols[n],est)
            if (v):
                row[n] = round(v,sigfigs) #v
            else:
                row[n] = 0.0
#           row[n] = np.round(get_estimate(tt_cols[n],est),3)
#           row[n] = round(get_estimate(tt_cols[n],est),decimals=3,notation='sci')

        tt = np.append(tt,np.array([row]),axis=0)
        func = np.append(func,round(       get_metadata('fn',meta),sigfigs))
        v =  np.exp(get_estimate('loggamma',est))
        gamm = np.append(gamm,round(float(v),sigfigs=4)) #,notation='sci'))

    tt = pd.DataFrame(tt,columns=tt_cols)#,dtype=float)
    tt['fn'] = func
#   tt['gamma'] = gamm
    print(tt)

    tex = fit_path+'fit_table.tex'
#                       float_format='%.3f'
    tt.to_latex(buf=tex,index=False)
#   print('Table written to file',tex)
    #      '. \usepackage{booktabs} is required.')

###################################################   
###################################################   
###################################################   
###################################################   
if __name__ == '__main__':
#   make_ADMB_dat(county_dat,'San Francisco')
#   for c in BayAreaCounties:
#       make_ADMB_dat(county_dat,c)

#   plot_counties(county_dat,Counties=['Alameda'],
#                 death_threshold=1, cases_threshold=10,file='LA')
#   plot_counties(county_dat,Counties=LargestCACounties,
#                 death_threshold=1, cases_threshold=10,
#                 per_capita=False,yscale='log',file='BigCA')
#   plot_counties(county_dat,Counties=BayAreaCounties,
#                 death_threshold=1, cases_threshold=10, file='BayArea')
#   plot_counties(county_dat,Counties=NewYorkCounties,
#                 death_threshold=5, cases_threshold=50)
#   plot_counties(county_dat,Counties=['Orange'],
#                 death_threshold=1, cases_threshold=10,file='Orange')
#   plot_beta_mu(['Alameda','Santa_Clara' ])
#   plot_beta_mu([
#                 'Alameda',
#                 'Contra_Costa',
#                 'San_Francisco',
#                 'San_Mateo',
#                 'Santa_Clara'
#                 ]
#                 ,delta_ts=True
#                 )
#   make_fit_table(['Alameda','Contra_Costa','San_Francisco','San_Mateo','Santa_Clara'])
    for c in ['Alameda','Contra_Costa','San_Francisco','San_Mateo','Santa_Clara']:
        plot_diagnostics(c)
#   plot_diagnostics('Alameda')
else:
    print('type something')

