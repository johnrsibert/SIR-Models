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
fit_path = '/home/jsibert/Projects/SIR-Models/fits/'

FirstNYTDate = datetime.strptime('2020-01-21','%Y-%m-%d')
CAOrderDate = datetime.strptime('2020-03-19','%Y-%m-%d')
EndOfTime = datetime.strptime('2020-05-31','%Y-%m-%d')

def Strptime(x):
   """
   wrapper for datetime.strptime callable by map(..)
   """
   s = str(x)
   y = datetime.strptime(x,'%Y-%m-%d')
   return(y)

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

        
def make_ADMB_dat(dat,County):
    """ Generate input data for analysis by ADMB or TMB models
        dat: pandas object read by another function
        County: string containing the name of the county in California
    """

    maxdt = check_delta_t(dat,County)
    if (maxdt > 1):
        print('WARNING: timstep greater than one for',County,'County')

    State = 'California'
    pops = pd.read_csv(pop_data_path,header=0)        # get county population
    cp_row = pops['county'] == County 
    population = int(pops[cp_row]['population'])

    mtime = os.path.getmtime("/home/other/nytimes-covid-19-data/us-counties.csv")
    dtime = datetime.fromtimestamp(mtime)
    update_stamp = str(dtime.date())
 

#   print(dat.head())
#   get county cases & deaths    
    state_filter = dat['state'].isin([State])
    county_filter = dat['county'].isin([County])
    County_rows = state_filter & county_filter

    O = open(County.replace(' ','_')+'.dat','w')
    O.write('# county\n %s\n'%County.replace(' ','_'))
    O.write('# updateed from https://github.com/nytimes/covid-19-data.git\n')
    O.write(' %s\n'%update_stamp)
    O.write('# population (N0)\n %10d\n'%population)
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

def plot_county_fit(county,
                  death_threshold=1, cases_threshold=1,
                  yscale='linear', per_capita=False, delta_ts=False,
                  text_spacer='  ', file = None):
    """ 
    Plot predicted & observed cases and deaths vs time from threshold
    Counties: list of California counties to plotted
    """

    firstDate = mdates.date2num(FirstNYTDate)
    orderDate = mdates.date2num(CAOrderDate)
    lastDate  = mdates.date2num(EndOfTime)
    print(CAOrderDate,orderDate)

    date_list = pd.date_range(start=firstDate,end=lastDate)
    date_lim = [date_list[0],date_list[len(date_list)-1]]

    fig, ax = plt.subplots(2,1,figsize=(6.5,6.0))
    if (per_capita):
        ax[0].set_ylabel('Cases'+' per '+str(mult))
        ax[1].set_ylabel('Deaths'+' per '+str(mult))
    else:
        ax[0].set_ylabel('Cases')
        ax[1].set_ylabel('Deaths')


    for a in range(0,len(ax)):
        ax[a].set_xlim([firstDate,lastDate])
        ax[a].xaxis.set_major_formatter(dates.DateFormatter('%b'))
        ax[a].xaxis.set_major_locator(plt.MultipleLocator(30))
        ax[a].xaxis.set_minor_locator(plt.MultipleLocator(1))
#       if (delta_ts):
#           ax2.append(ax[a].twinx())
#   ax[1].xaxis.set_minor_locator(plt.MultipleLocator(1))


#   ax[0].set_xlabel('Time (days after n>%.0f)'%cases_threshold)
#   ax[1].set_xlabel('Time (days after n>%.0f)'%death_threshold)

    pn = fit_path+county.replace(' ','_')+'.RData'
    fit=pyreadr.read_r(pn)
    diag = fit['diag']
    t = diag.index
    print(t)

    obsI = np.exp(diag['log_obs_cases'])
    preI = np.exp(diag['log_pred_cases'])
    obsD = np.exp(diag['log_obs_deaths'])
    preD = np.exp(diag['log_pred_deaths'])

    sigma_logC = 0.09
    sigma_logD = 0.06

    ax[0].set_yscale(yscale)
    plot_error(ax[0],t,diag['log_pred_cases'],sigma_logC)
    ax[0].scatter(t,obsI)
    ax[0].plot(t,preI,linewidth=2,color='red')

    ax[0].set_yscale(yscale)
    plot_error(ax[1],t,diag['log_pred_deaths'],sigma_logD)
    ax[1].scatter(t,obsD)
    ax[1].plot(t,preD,linewidth=2,color='red')

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
    pops = pd.read_csv(pop_data_path,header=0)
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

#   date,county,state,fips,cases,deaths
    print(dat.columns)
    for cc in Counties:
        state_filter = dat['state'].isin(['California'])
        county_filter = dat['county'].isin([cc])
    #   threshold_filter = dat[Column] > threshold
        County_rows = state_filter & county_filter# & threshold_filter
        cdata = dat[County_rows]
        print('cdata:',cdata.shape)
        county = cc.replace(' ','_')
        County = cc
        

        if (per_capita):
            cp_row = pops['county'] == County
            pop_size = int(pops[cp_row]['population'])
            cases =  mult*cdata['cases']/pop_size + eps
            deaths =  mult*cdata['deaths']/pop_size + eps
#           cdata['deaths'] = mult*cdata['deaths']/pop_size + eps
#           cdata['cases'] = mult*cdata['cases']/pop_size + eps
        else :
            cases =  cdata['cases']
            deaths =  cdata['deaths']
        
        Date = []
        for d in cdata['date']:
            Date.append(mdates.date2num(datetime.strptime(d,'%Y-%m-%d').date()))

        
        c = ax[0].plot(Date, cases,label=County)
        if (delta_ts):
            delta_cases = cases.diff()
            ax2[0].bar(Date,delta_cases,alpha=0.5)

        d = ax[1].plot(Date, deaths,label=County)
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
    return( meta.data[r].values[0])

def get_estimate(ename, ests):
    r = ests['names'].isin([ename])
    if (r.any() == True):
        return(float(ests.est[r]))
    else:
        return(None)

def get_objpar(pname, ests):
    r = ests['names'].isin([pname])
    if (r.any() == True):
        return(float(ests.obs[r]))
    else:
        return(None)
   
def plot_error(ax,x,logy,logsdy,mult=2.0):
    sdyu = np.array(np.exp(logy + mult*logsdy))
    sdyl = np.array(np.exp(logy - mult*logsdy))
    xy = np.array([x,sdyu])
    xy = np.append(xy,np.array([np.flip(x,0),np.flip(sdyl,0)]),axis=1)
    xp = np.transpose(xy).shape
    sd_region = plt.Polygon(np.transpose(xy), alpha=0.5,
                            facecolor='0.9', edgecolor='0.5')
    ax.add_patch(sd_region)
   
#def plot_error(ax,x,logy,logsdy,mult=2.0):
#    sdyu = np.array(np.exp(logy + mult*logsdy))
#    sdyl = np.array(np.exp(logy - mult*logsdy))
#    xy = np.array([x,sdyu])
#    xy = np.append(xy,np.array([np.flip(x,0),np.flip(sdyl,0)]),axis=1)
#    xp = np.transpose(xy).shape
#    sd_region = plt.Polygon(np.transpose(xy), alpha=0.5,
#                            facecolor='0.9', edgecolor='0.5')
#    ax.add_patch(sd_region)

def plot_beta_mu(fit_files=['Alameda'],delta_ts=False):
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
        est  = fit['ests']
        Date0 = get_metadata('Date0',meta)
        Date0 = datetime.strptime(Date0,'%Y-%m-%d')

        ntime = int(get_metadata('ntime',meta))
        tt = list(range(ntime))
        beta = np.exp(diag['logbeta'])
        mu = np.exp(diag['logmu'])

        pdate = []
        for t in tt:
            pdate.append(mdates.date2num(Date0 + timedelta(days=t)))
#  produces: [737487.0, 737488.0, 737489.0, ... , 737559.0, 737560.0, 737561.0]

        ax[0].plot(pdate,beta,label = fn)
        sigma_logbeta = get_estimate('sigma_logbeta',est)
        plot_error(ax[0],pdate,diag['logbeta'],sigma_logbeta)
        if (delta_ts):
            delta_obs_cases = diag['obs_cases'].diff()
            ax2[0].bar(pdate,delta_obs_cases,alpha=0.5,label=fn)
            ax2[0].set_ylabel('Cases')

        ax[1].plot(pdate,mu,label = fn)
        sigma_logmu = get_estimate('sigma_logmu',est)
        plot_error(ax[1],pdate,diag['logmu'],sigma_logmu)
        if (delta_ts):
            delta_obs_deaths = diag['obs_deaths'].diff()
            ax2[1].bar(pdate,delta_obs_deaths,alpha=0.5,label=fn)
            ax2[1].set_ylabel('Deaths')

    for a in range(0,len(ax)):
    #   Adjust length of y axis
        ax[a].set_ylim(0,ax[a].get_ylim()[1])
    #   Newsome's shelter in place order
        ax[a].plot((orderDate,orderDate),
                   (ax[a].get_ylim()[0], ax[a].get_ylim()[1]),color='black')
        ax[a].legend()

    plt.show()

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
                 fit_path = '/home/jsibert/Projects/SIR-Models/fits/',
                 save = False):
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

def make_fit_table(fit_files=['Alameda','Santa_Clara'],
                 fit_path = '/home/jsibert/Projects/SIR-Models/fits/'):
    tt_cols = ['county','fn'  ,'sigma_logP'   ,'sigma_logbeta'  ,'sigma_logmu',
               'loggamma'      ,'gamma']
#   header  = ['County',r'$f$',r'$\sigma_\nu$',r'$\sigma_\beta$',r'$\sigma_\mu$',
#              r'$\log~\gamma$',r'$\gamma$'
#              r'$\tilde{\beta}$',r'$\tilde{\mu}$']
#   tt_cols = ['county','fn','sigma_logP','sigma_logbeta','sigma_logmu']
    tt = pd.DataFrame(columns=tt_cols,dtype=None)

    func = pd.DataFrame(columns=['fn'])
    gamm = pd.DataFrame(columns=['gamma'])
    mbeta = pd.DataFrame(columns=['mbeta'])
    mmu = pd.DataFrame(columns=['mmu'])
    sigfigs = 3
    for fn in fit_files:
        pn = fit_path+fn+'.RData'
        fit=pyreadr.read_r(pn)
        ests  = fit['ests']
        meta = fit['meta']
        diag = fit['diag']
        row = [None]*len(tt_cols)
        row[0] = fn
        for n in range(1,len(tt_cols)):
            v = get_estimate(tt_cols[n],ests)
            if (v):
                row[n] = round(v,sigfigs) #v
            else:
                v = get_objpar(tt_cols[n],ests)
#               if (v):
#                  row[n] = round(v,sigfigs) #v
#               else:
#                  row[n] = 0.0
#           row[n] = np.round(get_estimate(tt_cols[n],est),3)
#           row[n] = round(get_estimate(tt_cols[n],est),decimals=3,notation='sci')

        tt = np.append(tt,np.array([row]),axis=0)
        func = np.append(func,round(       get_metadata('fn',meta),sigfigs))
        v =  np.exp(get_estimate('loggamma',ests))
        gamm = np.append(gamm,round(float(v),sigfigs=4)) #,notation='sci'))
        beta = np.exp(diag['logbeta'])
        mbeta = np.append(mbeta,round(float(beta.quantile(q=0.5)),sigfigs))
        mu = np.exp(diag['logmu'])
        mmu = np.append(mmu,round(float(mu.quantile(q=0.5)),sigfigs))

    tt = pd.DataFrame(tt,columns=tt_cols)#,dtype=float)
    tt['fn'] = func
    tt['gamma'] = gamm
    tt['mbeta'] = mbeta
    tt['mmu'] = mmu
    print(tt)

#   print(tt.shape)
#   print(len(header))
    tex = fit_path+'fit_table.tex'
#                       float_format='%.3f'
    tt.to_latex(buf=tex,index=False)#,header=header)
#   print('Table written to file',tex)
    #      '. \usepackage{booktabs} is required.')

##############################################################################
##############################################################################
if __name__ == '__main__':

#   plot_county_fit('Riverside')

#   make_ADMB_dat(county_dat,'San Francisco')

#   for c in LargestCACounties:
#       make_ADMB_dat(county_dat,c)

    plot_county_dat(county_dat,
                  Counties=['San Bernardino','San Diego','San Francisco'],
                  death_threshold=1, cases_threshold=1,file='county_plot',
                  delta_ts=True,per_capita=True)

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
#   plot_beta_mu(['Riverside']) #,'San_Mateo' ])#,delta_ts=True)
#   plot_beta_mu([ 'Contra_Costa', ] ,delta_ts=True)

#   make_fit_table(['Alameda','Contra_Costa','San_Francisco','San_Mateo','Santa_Clara'])
#   make_fit_table([ "Alameda", "Contra_Costa", "Los_Angeles", "Marin",
#                      "Napa", "Orange", "Riverside", "Sacramento",
#                      "San_Bernardino", "San_Diego", "San_Francisco",
#                      "San_Mateo", "Santa_Clara", "Sonoma"])

#   for c in ['Alameda','Contra_Costa','San_Francisco','San_Mateo','Santa_Clara']:
#       plot_diagnostics(c)
#   plot_diagnostics('Alameda')


#   for c in LargestCACounties:
#       check_delta_t(county_dat,c)

else:
    print('type something')

