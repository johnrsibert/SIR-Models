#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taking a more OO approach to the code in cv19.py

Created on Thu Jul  2 09:04:11 2020

@author: jsibert
"""

import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc
import numpy as np
import os
import sys
import pyreadr
from io import StringIO
import scipy.stats as stats
from sigfig import round
from tabulate import tabulate
from collections import OrderedDict
import glob
import re

plt.style.use('file:///home/jsibert/.config/matplotlib/john.mplstyle')
import js_covid as cv

eps = 1e-5

# ---------------- global utility functions ---------------------------

def Strptime(x):
    """
    wrapper for datetime.strptime callable by map(..)
    """
    y = datetime.strptime(x,'%Y-%m-%d')
    return(y)

def prop_scale(lim,prop):
    s = lim[0] + prop*(lim[1]-lim[0])
    return(s)
    
def pretty_county(s):
    ls = len(s)
    pretty = s[0:(ls-2)]+', '+s[(ls-2):]
    return(pretty.replace('_',' ',5))

def short_name(s):
    """
    Create 4 byte abbreviation for getography names
    """
    w = re.split(r'[ _-]',s)
    if (len(w)<2):
        sn = s[0:2]+s[-2:]
    else:
        sn = w[0][0]+w[1][0]+s[-2:]
    return(sn)  

def mark_ends(ax,x,y,label,end='b',spacer=' '):
    c = ax.get_lines()[-1].get_color()
    a = ax.get_lines()[-1].get_alpha()
#   print('color, alpha:',c,a)
    if ( (end =='l') | (end == 'b')):
        mark = ax.text(x[0],y[0],label+spacer,ha='right',va='center',fontsize=8,
                color=c) #,alpha=a)

    if ( (end =='r') | (end == 'b')):
        i = len(x)-1
        mark = ax.text(x[i],y[i],spacer+label,ha='left',va='center',fontsize=8,
                color=c) #,alpha=a)
                      # Set the alpha value used for blendingD - 
    mark.set_alpha(a) # not supported on all backends

def plot_error(ax,x,y,sdy,logscale=True,mult=2.0):
    # use of logscale is counterintuitive;
    # indicates that the y variable is logarithmic
    if (logscale):
        sdyu = np.array(y + mult*sdy)
        sdyl = np.array(y - mult*sdy)
    else:
        sdyu = np.array(np.exp(y + mult*sdy))
        sdyl = np.array(np.exp(y - mult*sdy))

    xy = np.array([x,sdyu])
    xy = np.append(xy,np.array([np.flip(x,0),np.flip(sdyl,0)]),axis=1)
    xp = np.transpose(xy).shape
    c = ax.get_lines()[-1].get_color()
    sd_region = plt.Polygon(np.transpose(xy), alpha=0.5,
                            facecolor='0.9', edgecolor=c,lw=1)
    ax.add_patch(sd_region)


def isNaN(num):
    return num != num

def median(x):
    mx = x.quantile(q=0.5)
    return float(mx)

def add_order_date(ax):
#   Newsome's shelter in place order
    orderDate = mdates.date2num(cv.CAOrderDate)
    ax.plot((orderDate,orderDate),
            (ax.get_ylim()[0], ax.get_ylim()[1]),
    #       (0, ax.get_ylim()[1]),
            color='0.5', linewidth=3,alpha=0.5)

# -----------  class definitions--------------       

class Geography:

#   def __init__(self,name,enclosed_by,code):
    def __init__(self,**kwargs):
        self.gtype = None
        self.name = kwargs.get('name')
        self.enclosed_by = kwargs.get('enclosed_by')
        self.code = kwargs.get('code')
        self.population = None
        self.source = None
        if self.name is None:
            self.moniker = None
        else:
            self.moniker = self.name+self.code
            self.moniker =  self.moniker.replace(' ','_',5) 
    #   self.TMB_fit = None
    #   self.ADMB_fit = None
        if self.moniker is None:
            self.dat_file = None
        else:
            self.dat_file = cv.dat_path+self.moniker+'.dat'
        self.updated = None
        self.date0 = None
        self.ntime = None
        self.date = None
        self.cases = None
        self.deaths = None
        self.pdate = None # date for plotting on x-axis
        
    def print(self):
        print(self)
        
        
    def print_metadata(self):
        print('gtype:',self.gtype)
        print('gname:',self.name)
        print('genclosed_by:',self.enclosed_by)
        print('gcode:',self.code)
        print('gpopulation:',self.population)
        print('gsource:',self.source)
        print('gmoniker:',self.moniker)
        print('gdat_file:',self.dat_file)
        print('gupdated:',self.updated)
        print('gdate0:',self.date0)

    def print_data(self):
        print('date','cases','deaths','pdate')
        for k in range(0,len(self.date)):
            print(self.date[k],self.cases[k],self.deaths[k],self.pdate[k])
            
    def get_pdate(self):
        if (self.pdate != None):
            return(self.pdate)

        else:
            self.pdate = []
            for d in self.date:
                self.pdate.append(mdates.date2num(datetime.strptime(d,'%Y-%m-%d').date()))
            return(self.pdate)

    def get_county_pop(self):
        """
        Reads US Census populations estimates for 2019
        Subset of orginal csv file saved in UTF-8 format. 
        The word 'County' stipped from county names.

        nyc_pop is the sum of census estimates for New York, Kings, Queens
        Bronx, and Richmond counties to be compatible with NYTimes data
        """
        
        nyc_pop = 26161672
        if (self.name == 'New York City'):
            return(nyc_pop)

        dat = pd.read_csv(cv.census_data_path,header=0)
        state_filter = dat['state'].isin([self.enclosed_by])
        county_filter = dat['county'].isin([self.name])
        COUNTY_filter = (dat['COUNTY']>0)
        County_rows = state_filter & county_filter & COUNTY_filter
        try:
            population = int(pd.to_numeric(dat[County_rows]['population'].values))
        except:
            print('get_county_pop() failed for:')
            print(self.name, self.enclosed_by,self.code) 
            population = 1
        return(population)   
        
    def read_nyt_data(self,gtype='county'):
        self.gtype = gtype
        if (gtype == 'county'):
            cspath = cv.NYT_counties
        else:
            sys.exit('No data for',gtype)
        
        
        dat = pd.read_csv(cspath,header=0)

        mtime = os.path.getmtime(cspath)
        dtime = datetime.fromtimestamp(mtime)
        self.updated = str(dtime.date())
        self.population = self.get_county_pop()
    
        state_filter = dat['state'].isin([self.enclosed_by])
        county_filter = dat['county'].isin([self.name])
        County_rows = state_filter & county_filter
        if (len(County_rows) < 1):
            sys.exit(' * * * no recores fround for',self.name,self.surrounded_by)

   #    tmp = dat[County_rows]['date'].map(Strptime)
   #    self.pdate  = np.array(mdates.date2num(tmp))
        self.cases  = np.array(dat[County_rows]['cases'])
        self.deaths = np.array(dat[County_rows]['deaths'])
        self.date   = np.array(dat[County_rows]['date'])
        self.date0 = self.date[0]
        self.ntime = len(self.date)
        
    def write_dat_file(self):
        print(len(self.date),'records found for',self.name,self.enclosed_by)
        O = open(self.dat_file,'w')
        O.write('# county\n %s\n'%(self.moniker))
        O.write('# updated from https://github.com/nytimes/covid-19-data.git\n')
        O.write(' %s\n'%self.updated)
        O.write('# population (N0)\n %10d\n'%self.population)
        O.write('# date zero\n %s\n'%self.date0)
        ntime = len(self.date)-1
        O.write('# ntime\n %4d\n'%ntime)
        O.write('#%6s %5s\n'%('cases','deaths'))
        for r in range(0,len(self.date)):
            O.write('%7d %6d\n'%(self.cases[r],self.deaths[r]))

    def dow_count(self,mult = 1000.0):
        """
        Accumulate first differences of cases and deaths by day of week
        Scale by mult/population
        """
        names = ['Mo','Tu','We','Th','Fr','Sa','Su','moniker']
        Ccount = pd.Series(0.0,index=names)
        Dcount = pd.Series(0.0,index=names)
        d1_cases = np.diff(self.cases)
        d1_deaths = np.diff(self.deaths)
    
        for k in range(0,len(self.date)-1):            
            j = datetime.strptime(self.date[k],'%Y-%m-%d').weekday()
            Ccount[j] += d1_cases[k]
            Dcount[j] += d1_deaths[k]

        scale = mult/self.population
        Ccount = Ccount * scale
        Dcount = Dcount * scale
        Ccount['moniker'] = self.moniker
        Dcount['moniker'] = self.moniker
        counts = [Ccount,Dcount]
        return(counts)


    def plot_prevalence(self,yscale='linear', per_capita=False, delta_ts=True,
                        window=[11], plot_dt = False, annotation = True,
                        signature = False, save = True):
        
        """ 
        Plots cases and deaths vs calendar date 

        scale: select linear of log scale 'linear'
        per_capita: plot numbers per 1000 (see mult)  False
        delta_ts: plot daily new cases True
        window: plot moving agerage window [11]
        plot_dt: plot initial doubling time slopes on log scale False
        annotations: add title and acknowledgements True
        save : save plot as file
        """
        mult = 1000
    
        firstDate = mdates.date2num(cv.FirstNYTDate)
        orderDate = mdates.date2num(cv.CAOrderDate)
        lastDate  = mdates.date2num(cv.EndOfTime)
    
        fig, ax = plt.subplots(2,1,figsize=(6.5,4.5))
        if (per_capita):
            ax[0].set_ylabel('Daily Cases'+' per '+str(mult))
            ax[1].set_ylabel('Daily Deaths'+' per '+str(mult))
        else:
            ax[0].set_ylabel('Daily Cases')
            ax[1].set_ylabel('Daily Deaths')
    
    
        ax2 = []
        for a in range(0,len(ax)):
            self.make_date_axis(ax[a])

            ax[a].set_yscale(yscale)
            ax2.append(ax[a].twinx())
            ax2[a].set_ylabel('Cumulative')
        
        if (per_capita):
            cases =  mult*self.cases/self.population + eps
            deaths =  mult*self.deaths/self.population + eps
        else :
            cases  =  self.cases
            deaths =  self.deaths
        
        nn = self.ntime-1

        Date = self.get_pdate()

        delta_cases = np.diff(cases)
        ax[0].bar(Date[1:], delta_cases)

 #      aug1 = mdates.date2num(datetime.strptime('2020-08-01','%Y-%m-%d').date())
 #      ax[0].plot((aug1,aug1),ax[0].get_ylim(),color='black',linewidth=1)

        for w in range(0,len(window)):
            adc = pd.Series(delta_cases).rolling(window=window[w]).mean()
            ax[0].plot(Date[1:],adc,linewidth=2)
            mark_ends(ax,Date,adc, str(window[w])+'da','r')
        
        ax2[0].plot(Date, cases,alpha=0.5, linewidth=1)#,label=cc)
        mark_ends(ax2,Date,cases,r'$\Sigma$C','r')

        if ((yscale == 'log') & (plot_dt)):
            ax[0] = self.plot_dtslopes(ax[0])
                
        ax2[1].plot(Date, deaths,alpha=0.5,linewidth=1)#,label=cc)
        mark_ends(ax2[1],Date,deaths,r'$\Sigma$D','r')

        delta_deaths = np.diff(deaths)
        ax[1].bar(Date[1:],delta_deaths)
        for w in range(0,len(window)):
            add = pd.Series(delta_deaths).rolling(window=window[w]).mean()
            ax[1].plot(Date[1:],add,linewidth=2)
            mark_ends(ax[1],Date,add, str(window[w])+'da','r')
    
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
    
        if (annotation):
            title = 'Covid-19 Prevalence in '+self.name+' County, '+ self.enclosed_by
            fig.text(0.5,1.0,title ,ha='center',va='top')
            fig.text(0.0,0.0,' Data source: New York Times, https://github.com/nytimes/covid-19-data.git.',
                     ha='left',va='bottom', fontsize=8)
    
            mtime = os.path.getmtime("/home/other/nytimes-covid-19-data/us-counties.csv")
            dtime = datetime.fromtimestamp(mtime)
            fig.text(1.0,0.0,'Updated '+str(dtime.date())+' ', ha='right',va='bottom', fontsize=8)

        if (signature):
            by_line = 'Graphics by John Sibert (https://github.com/johnrsibert/SIR-Models/tree/master/PlotsToShare) '
        #   fig.text(0.025,0.500,by_line+' ', ha='left',va='top', fontsize=8,alpha=0.25)#,color='red')
            fig.text(1.0,0.025,by_line+' ', ha='right',va='bottom', fontsize=8,alpha=0.25)#,color='red')
    
        if save:
            gfile = cv.graphics_path+self.moniker+'_prevalence.png'
            plt.savefig(gfile,dpi=300)
        #   plt.savefig('fig.png',bbox_inches='tight')
            plt.show(False)
            plt.pause(3)
            plt.close()
            
            print('plot saved as',gfile)
        else:
            plt.show()


    def make_date_axis(self,ax):
        firstDate = mdates.date2num(cv.FirstNYTDate)
        lastDate  = mdates.date2num(cv.EndOfTime)
        ax.set_xlim([firstDate,lastDate])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.DayLocator())


    def plot_dtslopes(self, ax, threshold = 2, dt = [1,2,4,8]):
        """
        Superimpose exponential growth slope lines for different doubling times
        ax: axisis on which to draw slopes lines
        threshold: number of cases used to start slopes
        dt: representative doubling times in days
        """
        k0 = 0
        # find frist date in shich cases exceeds threshold
        for k in range(0, self.ntime):
            if (self.cases[k] >= threshold):
                k0 = k
                break

        d0 = self.pdate[k0]
        c0 = self.cases[k0]
        sl = np.log(2.0)/dt
        xrange = ax.get_xlim()
        yrange = [25,ax.get_ylim()[1]]
        for i in range(0,len(dt)):
            y = c0 + np.exp(sl[i]*(d0-xrange[0]))
            ax.plot([d0,xrange[1]],[c0,y],color='black',linewidth=1)
            c = ax.get_lines()[-1].get_color()
            mark_ends(ax,xrange,y,str(dt[i])+' da','r')

        return(ax)

    def plot_per_capita_curvature(self,mult = 1000,save=False):
        pc_cases = self.cases/self.population * mult
        
        fig, ax = plt.subplots(1,figsize=(6.5,4.5))
        self.make_date_axis(ax)
        
        grad = np.gradient(pc_cases)
    #   d1 = np.diff(pc_cases)
    #   print(len(d1))
    #   grad = np.diff(d1)
    #   print(len(grad))

    #   ax.plot(self.get_pdate()[2:],grad)
        ax.plot(self.get_pdate(),grad)
        print(min(grad),max(grad))

        plt.show()




class Fit(Geography):

    def __init__(self,fit_file,**kwargs):
        super().__init__(**kwargs)
        pn = fit_file #cv.fit_path+fit_file
        filename, extension = os.path.splitext(pn)
        if (extension == '.RData'):
            self.fit_type = 'TMB'
            tfit=pyreadr.read_r(pn)
        else:
            sys.exit('class Fit  not yet implemented on '+extension+' files.')
        head, tail = os.path.split(pn)
        self.moniker = os.path.splitext(tail)[0]

        self.diag = tfit['diag']
        self.md = tfit['meta']
        self.ests = tfit['ests']
        self.date0 = self.get_metadata_item('Date0')
        self.ntime = int(self.get_metadata_item('ntime'))
    #   Date0 = self.get_metadata_item('Date0')
    #   print('Date0:',Date0)

    def print_metadata(self):
        super().print_metadata()
        print('type:',self.fit_type)

    def get_metadata_item(self,mdname):
        r = self.md['names'].isin([mdname])
        return(self.md.data[r].values[0])

    def get_estimate_item(self, ename):
        r = self.ests['names'].isin([ename])
        if (r.any() == True):
            return(float(self.ests.est[r]))
        else:
            return(None)
    
    def get_est_or_init(self,name):
        v = self.get_estimate_item(name) 
        if (isNaN(v)):
            v = self.get_initpar_item(name)
            return(v)
        else:
            return(v)
    
    def get_initpar_item(self,pname):
        r = self.ests['names'].isin([pname])
        if (r.any() == True):
            return(float(ests.init[r]))
        else:
            return(None)
       
    def plot(self,logscale=True, per_capita=False, delta_ts=False,
             npl = 4, save = True):
        """ 
        """
    
        firstDate = mdates.date2num(cv.FirstNYTDate)
        orderDate = mdates.date2num(cv.CAOrderDate)
        lastDate  = mdates.date2num(cv.EndOfTime)
    
        date_list = pd.date_range(start=firstDate,end=lastDate)
        date_lim = [date_list[0],date_list[len(date_list)-1]]
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams["scatter.marker"] = '+'
        plt.rcParams["lines.markersize"] = 6
        prefix = ''
        if (logscale):
            prefix = 'ln '
             
        fig, ax = plt.subplots(npl,1,figsize=(9.0,(npl)*3.0))
        if (per_capita):
            ax[0].set_ylabel(prefix+'Cases'+' per '+str(mult))
            ax[1].set_ylabel(prefix+'Deaths'+' per '+str(mult))
        else:
            ax[0].set_ylabel(prefix+'Cases')
            ax[1].set_ylabel(prefix+'Deaths')
        if (npl > 2):
            ax[2].set_ylabel(r'$\beta\ (da^{-1})$')
        if (npl > 3):
            ax[3].set_ylabel(r'$\mu\ (da^{-1})$')
    
    
        for a in range(0,len(ax)):
            self.make_date_axis(ax[a])
    
        Date0 = self.get_metadata_item('Date0')
        Date0 = datetime.strptime(Date0,'%Y-%m-%d')
        pdate = []
        for t in range(0,len(self.diag.index)):
            pdate.append(mdates.date2num(Date0 + timedelta(days=t)))
    
        obsI = self.diag['log_obs_cases']
        preI = self.diag['log_pred_cases']
        obsD = self.diag['log_obs_deaths']
        preD = self.diag['log_pred_deaths']
        sigma_logC = np.exp(self.get_est_or_init('logsigma_logC'))
        sigma_logD = np.exp(self.get_est_or_init('logsigma_logD'))
        sigma_logbeta = self.get_est_or_init('logsigma_logbeta')
        sigma_logmu = np.exp(self.get_est_or_init('logsigma_logmu'))
        if (logscale):
            ax[0].set_ylim(0.0,1.2*max(obsI))
            ax[0].scatter(pdate,obsI)
            ax[0].plot(pdate,preI,color='red')
        else:
            ax[0].set_ylim(0.0,1.2*max(np.exp(obsI)))
            ax[0].scatter(pdate,np.exp(obsI))
            ax[0].plot(pdate,np.exp(preI),color='red')
        plot_error(ax[0],pdate,obsI,sigma_logC,logscale)
        tx = prop_scale(ax[0].get_xlim(), 0.05)
        ty = prop_scale(ax[0].get_ylim(), 0.90)
        sigstr = '%s = %.3g'%('$\sigma_I$',sigma_logC)
        ax[0].text(tx,ty,sigstr, ha='left',va='center',fontsize=10)
    
        if (logscale):
            ax[1].set_ylim(0.0,1.2*max(obsD))
            ax[1].scatter(pdate,obsD)
            ax[1].plot(pdate,preD,color='red')
        else:
            ax[1].set_ylim(0.0,1.2*max(np.exp(obsD)))
            ax[1].scatter(pdate,np.exp(obsD))
            ax[1].plot(pdate,np.exp(preD),color='red')

        plot_error(ax[1],pdate,obsD,sigma_logD,logscale)
        tx = prop_scale(ax[1].get_xlim(), 0.05)
        ty = prop_scale(ax[1].get_ylim(), 0.90)
        sigstr = '%s = %.3g'%('$\sigma_D$',sigma_logD)
        ax[1].text(tx,ty,sigstr, ha='left',va='center',fontsize=10)
    
        if (npl > 2):
            log_beta = self.diag['logbeta']
        #   log_beta_ticks = [ 1.01978144,0.32663426,-0.36651292,
        #                     -1.0596601,-1.75280728,-2.44595446,
        #                     -3.13910164,-3.83224882,-4.525396  ]
        #   print(min(log_beta_ticks),max(log_beta_ticks))
        #   print(0.8*min(log_beta_ticks),1.2*max(log_beta_ticks))
        #   ax[2].set_ylim((1.2*min(log_beta)),1.2*max(log_beta))
        #   ax[2].set_yticks(log_beta_ticks)
            sigstr = '%s = %.3g'%('$\sigma_\\beta$',sigma_logbeta)
            tx = prop_scale(ax[2].get_xlim(), 0.05)
            ty = prop_scale(ax[2].get_ylim(), 0.90)
            ax[2].plot(pdate,log_beta)
            plot_error(ax[2],pdate,log_beta,sigma_logbeta,logscale=True)
            ax[2].text(tx,ty,sigstr, ha='left',va='center',fontsize=10)

            med = median(np.exp(log_beta))
            logmed = np.log(med)
            medstr = '%s = %.3g'%('$\\tilde{\\beta}$',med)
            ax[2].text(ax[2].get_xlim()[0],logmed,medstr,ha='left',va='bottom',fontsize=10)
            ax[2].plot(ax[2].get_xlim(),[logmed,logmed])

        #   increase frequcncy of tick marks
            start, end = ax[2].get_ylim()
            dtick = (end - start)/5
            ax[2].set_yticks(np.arange(start, end, dtick))


        #   finagle doubling time axis at same scale as beta
            add_doubling_time = False
            if (add_doubling_time):
                dtax = ax[2].twinx()
                dtax.set_ylim(ax[2].get_ylim())
                dtax.grid(False,axis='y') # omit grid lines
                dtax.set_ylabel('Doubling Time (da)')
            #   render now to get the tick positions and labels
                fig.show()
                fig.canvas.draw()
                y2_ticks = dtax.get_yticks()
                labels = dtax.get_yticklabels()
                for i in range(0,len(y2_ticks)):
                    y2_ticks[i] = np.log(2)/np.exp(y2_ticks[i])
                    if (y2_ticks[i] < 100.0):
                       labels[i] = '%.2g'%y2_ticks[i]
                    elif(y2_ticks[i] < 1000.0):
                       labels[i] = '>100'
                    else:
                       labels[i] = ''
                      
                dtax.tick_params(length=0)
                dtax.set_yticklabels(labels)
            #   fig.show()
            #   fig.canvas.draw()

        if (npl > 3):
            logmu = self.diag['logmu']
        #   ymin = min(self.diag['logmu'])-2.5*sigma_logmu
        #   ax[3].set_ylim(ymin,1.2*max(self.diag['logmu']))
        #   print(min(logmu),max(logmu))
        #   print(1.2*min(logmu),0.8*max(logmu))
            ax[3].set_ylim((1.2*min(logmu),0.8*max(logmu)))

            sigstr = '%s = %.3g'%('$\sigma_\\mu$',sigma_logmu)
            tx = prop_scale(ax[3].get_xlim(), 0.05)
            ty = prop_scale(ax[3].get_ylim(), 0.90)
            ax[3].plot(ax[2].get_xlim(),[0.0,0.0],color='0.2',linestyle='--')
            ax[3].plot(pdate,self.diag['logmu'])
            plot_error(ax[3],pdate,logmu,sigma_logmu,logscale=True)
            ax[3].text(tx,ty,sigstr, ha='left',va='center',fontsize=10)

            med = median(np.exp(self.diag['logmu']))
            logmed = np.log(med)
            medstr = '%s = %.3g'%('$\\tilde{\\mu}$',med)
            ax[3].text(ax[3].get_xlim()[0],logmed,medstr,ha='left',va='bottom',fontsize=10)
            ax[3].plot(ax[3].get_xlim(),[logmed,logmed])
    
        title = self.moniker #self.name+' County, '+self.enclosed_by
        fig.text(0.5,0.95,title ,ha='center',va='bottom')

        if save:
            gfile = cv.graphics_path+self.moniker+'_'+self.fit_type+'_estimates'
            if (not logscale):
                gfile = cv.graphics_path+self.moniker+'_'+self.fit_type+'_a'+'_estimates'
            fig.savefig(gfile+'.png')#,dpi=300)
            plt.show(False)
            print('plot saved as',gfile)
            plt.pause(2)
            plt.close()
        else:
            plt.show(True)



# ----------- end of class definitions--------------       

def get_mu_atD1(ext='.RData',fit_files = []):
    if (len(fit_files) < 1):
        fit_files = glob.glob(cv.fit_path+'*'+ext)
    else:
        for i,ff in enumerate(fit_files):
            fit_files[i] = cv.fit_path+fit_files[i]+ext

    mud_cols = ['moniker','lag','logmu','mu']
    tmp = np.empty(shape=(len(fit_files),4),dtype=object)
    for i,ff in enumerate(fit_files):
        moniker = os.path.splitext(os.path.basename(ff))[0]
        fit = Fit(ff)
        muD1 = -1.0
        obs_deaths = fit.diag['obs_deaths']
        for t in range(0,len(fit.diag.index)):
            if (muD1 < 0.0) and (obs_deaths[t] > 0.0):
                muD1 = np.exp(fit.diag['logmu'][t])
                lag = t-1
                tmp[i] = (moniker,lag,fit.diag['logmu'][t],muD1)

    mud = pd.DataFrame(tmp,columns=mud_cols)
    mud = mud.sort_values(by='lag',ascending=True)
    print(mud)
    mud.to_csv(cv.fit_path+'mud.csv',index=False)
    fig, ax = plt.subplots(1,figsize=(6.5,4.5))
    ax.set_ylabel('First '+r'$\ln\ \mu\ (da^{-1})$')
    ax.set_xlabel('Lag (days)') 
#   ax.hist(mud['logmu'],density=True,bins=3)
    ax.plot(mud['lag'],(mud['logmu']))
    plt.show(True)

def make_rate_plots(yvarname = 'logbeta',ext = '.RData', 
                    fit_files = [], show_medians = False, 
                    add_doubling_time = False, show_order_date = True, save=False):
    print(yvarname)
    if (yvarname == 'logbeta'):
        ylabel =r'$\ln \beta\ (da^{-1})$'
    elif (yvarname == 'logmu'):
        ylabel =r'$\ln \mu\ (da^{-1})$'
    elif (yvarname == 'gamma'):
        ylabel =r'$\gamma\ (da^{-1})$'
    else:
        sys.exit('Unknown yvarname '+yvarname)

    fig, ax = plt.subplots(1,figsize=(6.5,4.5))
    if (len(fit_files) < 1):
        suffix = '_g'
        fit_files = glob.glob(cv.fit_path+'*'+ext)
    #   fit_files = glob.glob(cv.fit_path+'constrainID/'+'*'+ext)
    else:
        suffix = ''
        for i,ff in enumerate(fit_files):
            fit_files[i] = cv.fit_path+fit_files[i]+ext
        #   fit_files[i] = cv.fit_path+'constrainID/'+fit_files[i]+ext
    for i,ff in enumerate(fit_files):
        print(i,ff)
        fit = Fit(ff)
        if (i == 0):
            fit.make_date_axis(ax)
            ax.set_ylabel(ylabel)
       
    #   fit.read_nyt_data()
        pdate = []
        Date0 = datetime.strptime(fit.date0,'%Y-%m-%d')
        for t in range(0,len(fit.diag.index)):
            pdate.append(mdates.date2num(Date0 + timedelta(days=t)))

    #   pdate = fit.get_pdate()

        yvar = fit.diag[yvarname]
        if (yvarname == 'gamma'):
            print(yvarname)
            print(min(yvar),max(yvar))
        ax.plot(pdate,yvar)

    #   aug1 = mdates.date2num(datetime.strptime('2020-08-01','%Y-%m-%d').date())
    #   ax.plot((aug1,aug1),ax.get_ylim(),color='black',linewidth=1)
      
    #   sigma_logbeta is the standard deviation of the generating
    #   random walk, NOT the standard deviation of the estimated
    #   random effect
    #   if (yvarname == 'logbeta' and len(fit_files) <=4):
    #       sigma_logbeta = fit.get_est_or_init('logsigma_logbeta')
    #       plot_error(ax,pdate,yvar,sigma_logbeta,logscale=True)

        sn = short_name(fit.moniker)
        mark_ends(ax,pdate,yvar,sn,'b')
        if (show_medians):
            med = median(np.exp(yvar))
            logmed = np.log(med)
            ax.plot(ax.get_xlim(),[logmed,logmed],linewidth=1,
                    color=ax.get_lines()[-1].get_color())

    if (show_order_date):
        add_order_date(ax)

#   finagle doubling time axis at same scale as beta
    if (add_doubling_time):
        yticks = np.arange(-7,1,1)
        ax.set_ylim(min(yticks)-1,max(yticks)+1)
        ax.set_yticks(yticks)
        dtax = ax.twinx()
        dtax.set_ylim(ax.get_ylim())
        dtax.grid(False,axis='y') # omit grid lines
        dtax.set_ylabel('Doubling Time (da)')
    #   render now to get the tick positions and labels
    #   fig.canvas.draw()
        y2_ticks = dtax.get_yticks()
        labels = dtax.get_yticklabels()
        for i in range(0,len(y2_ticks)):
            y2_ticks[i] = np.log(2)/np.exp(y2_ticks[i])
        #   labels[i] = '%.1f'%y2_ticks[i]
            labels[i] = round(float(y2_ticks[i]),2)

        dtax.tick_params(length=0)
        dtax.set_yticklabels(labels)

    if save:
        gfile = cv.graphics_path+yvarname+'_summary'+suffix+'.eps'
        fig.savefig(gfile)
        print('plot saved as',gfile)
        plt.show(True)
    else:
        plt.show(True)

def make_fit_plots(ext = '.RData'):
    fit_files = glob.glob(cv.fit_path+'*'+ext)
    print('found',len(fit_files),ext,'files in',cv.fit_path)
    plt.rc('figure', max_open_warning = 0)
    for ff in fit_files:
        fit = Fit(ff)
    #   fit.print_metadata()
        fit.plot(logscale=True)



def make_fit_table(ext = '.RData'):
    fit_files = glob.glob(cv.fit_path+'*'+ext)
    print('found',len(fit_files),ext,'files in',cv.fit_path)
    # mtime = os.path.getmtime(cspath) cv.NYT_counties
    #   dtime = datetime.fromtimestamp(mtime)
    #   self.updated = str(dtime.date())
    # updated from https://github.com/nytimes/covid-19-data.git

#   md_cols = ['county','N0','ntime','prop_zero_deaths','fn']
    md_cols = ['county','ntime','prop_zero_deaths','fn','C']
    es_cols = ['logsigma_logP'   , 'logsigma_logbeta'  , 'logsigma_logmu'  ,
               'logsigma_logC','logsigma_logD', 'mgamma','mbeta','mmu']
    tt_cols = md_cols + es_cols
    header = ['County','$n$','$p_0$','$f$','$C$',
              '$\sigma_\eta$','$\sigma_\\beta$','$\sigma_\\mu$',
              '$\sigma_I$','$\sigma_D$','$\\tilde\\gamma$','$\\tilde{\\beta}$','$\\tilde{\\mu}$']

    tt = pd.DataFrame(columns=tt_cols,dtype=None)

    func = pd.DataFrame(columns=['fn'])
    mgamma = pd.DataFrame(columns=['mgamma'])
    mbeta = pd.DataFrame(columns=['mbeta'])
    mmu = pd.DataFrame(columns=['mmu'])
    sigfigs = 3
#   for ff in fit_files:
    for k in range(0,len(fit_files)):
        ff = fit_files[k]
    #   fn = ff.replace(' ','_',5) 
    #   pn = fit_path+fn+'.RData'
    #   fit=pyreadr.read_r(pn)
        print('adding fit',k,ff)
        fit = Fit(ff)
        ests  = fit.ests #['ests']
        meta = fit.md #['meta']
        diag = fit.diag #['diag']
        row = pd.Series(index=tt_cols)
        county = fit.get_metadata_item('county')  
        row['county'] = pretty_county(county)
        for k in range(1,len(tt_cols)):
            n = tt_cols[k]
            v = fit.get_est_or_init(n)
            if ("logsigma" in n):
                if (v != None):
                    v = float(np.exp(v))
            row.iloc[k] = v

    #   row['N0'] = int(get_metadata('N0',meta))
        row['C'] = fit.get_metadata_item('convergence')
        row['ntime'] = int(fit.get_metadata_item('ntime'))
        row['prop_zero_deaths'] = round(float(fit.get_metadata_item('prop_zero_deaths')),sigfigs)
        tt = tt.append(row,ignore_index=True)
        func = np.append(func,float(fit.get_metadata_item('fn')))
        beta = np.exp(diag['logbeta'])
        mbeta = np.append(mbeta,median(beta))
        mu = np.exp(diag['logmu'])
        mmu = np.append(mmu,mu.quantile(q=0.5))
        gamma = diag['gamma']
        mgamma = np.append(mgamma,gamma.quantile(q=0.5))

    tt['fn'] = func
    tt['mgamma'] = mgamma
    tt['mbeta'] = mbeta
    tt['mmu'] = mmu

    tt = tt.sort_values(by='mbeta',ascending=True)#,inplace=True)

    for c in range(3,len(tt.columns)):
        for r in range(0,tt.shape[0]):
           if (tt.iloc[r,c] != None):
               tt.iloc[r,c] = round(float(tt.iloc[r,c]),sigfigs)

    row = pd.Series(None,index=tt.columns)
    row['county'] = 'Median'
    for n in tt.columns:
        if (n != 'county'):
            mn = tt[n].quantile()
            row[n] = mn
    tt = tt.append(row,ignore_index=True)

    mtime = os.path.getmtime(cv.NYT_counties)
    dtime = datetime.fromtimestamp(mtime)

    tex = cv.fit_path+'fit_table.tex'
    ff = open(tex, 'w')
    caption_text = "Model results. Estimating $\\beta$ and $\mu$ trends as random effects with computed $\gamma$.\nData updated " + str(dtime.date()) + " from https://github.com/nytimes/covid-19-data.git."

    ff.write(caption_text)
    ff.write(str(dtime.date())+'\n')
    ff.write(tabulate(tt, header, tablefmt="latex_raw",showindex=False))
#   tt.to_latex(buf=tex,index=False,index_names=False,longtable=False,
#               header=header,escape=False,#float_format='{:0.4f}'.format
#               na_rep='',column_format='lrrrrrrrrrrr')
    print('Fit table written to file',tex)

        
def make_dow_table(mult=1000):        
    """
    accumulate dow counts by geography as table rows
    """
    names = ['Mo','Tu','We','Th','Fr','Sa','Su']
    csdat = pd.read_csv(cv.large_county_path,header=0)
    cases_count  = pd.DataFrame(dtype=float)
    deaths_count = pd.DataFrame(dtype=float)
   
    for cs in range(0,len(csdat)):
        tmpG = Geography(csdat['county'][cs],csdat['state'][cs],csdat['ST'][cs])
        tmpG.read_nyt_data('county')
        row = tmpG.dow_count(mult)
    #   print(tmpG.moniker,type(row),len(row))
        cases_count  = cases_count.append(row[0],ignore_index=True)
        deaths_count = deaths_count.append(row[1],ignore_index=True)
      
    cases_count = cases_count.set_index('moniker')
    deaths_count = deaths_count.set_index('moniker')
    cases_count = cases_count.reindex(columns=names)
    deaths_count = deaths_count.reindex(columns=names)
    return[cases_count,deaths_count]
    
def plot_dow_boxes(mult=1000):
    counts = make_dow_table(mult)
    labels = counts[0].columns
    title = str(counts[0].shape[0])+' most populous US counties'

    fig, ax = plt.subplots(2,1,figsize=(6.5,4.5))
    fig.text(0.5,0.9,title,ha='center',va='bottom')
    ax[0].set_ylabel('Cases'+' per '+str(mult))
    ax[0].boxplot(counts[0].transpose(),labels=labels)
    ax[1].set_ylabel('Deaths'+' per '+str(mult))
    ax[1].boxplot(counts[1].transpose(),labels=labels)
  
    plt.savefig('days_of_week.png',dpi=300)
    plt.show()
  
def web_update():
    os.system('git -C /home/other/nytimes-covid-19-data pull -v')
    
    BC_cases_file = 'BCCDC_COVID19_Dashboard_Case_Details.csv'
    cmd = 'wget http://www.bccdc.ca/Health-Info-Site/Documents/' + BC_cases_file +\
         ' -O '+cv.cv_home+BC_cases_file
    print(cmd)
    os.system(cmd)

def make_dat_files():
#   gg = pd.read_csv(cv.cv_home+'UpdateList.csv',header=0,comment='#')
    gg = pd.read_csv(cv.cv_home+'top30.csv',header=0,comment='#')
    names = gg.columns
    gg = np.append(gg,np.array([
                                ['Honolulu','Hawaii','HI'],
                                ['Multnomah','Oregon','OR']
                               ]),axis=0)
    gg = pd.DataFrame(gg,columns=names)
#   plt.rc('figure', max_open_warning = 0)
    for g in range(0,len(gg)):
        print(gg['name'][g])
        tmpG = Geography(name=gg['name'][g], enclosed_by=gg['enclosed_by'][g],
                         code=gg['code'][g])
        tmpG.read_nyt_data('county')
        tmpG.write_dat_file()
        tmpG.plot_prevalence(save=True,per_capita=False)

def update_fits():
    save_wd = os.getcwd()
    print('save:',save_wd)
    print(cv.TMB_path)
    os.chdir(cv.TMB_path)
    print('current',os.getcwd())
    # globs s list of counties in runSS4.R
    cmd = 'Rscript --verbose simpleSIR4.R'
    print('running',cmd)
    os.system(cmd)
    os.chdir(save_wd)
    print('current',os.getcwd())

def update_shared_plots():
    update_list = pd.DataFrame(np.array(
       [['District of Columbia','District of Columbia','DC'],
        ['Honolulu','Hawaii','HI'],
        ['Multnomah','Oregon','OR'],
        ['Tompkins','New York','NY'],
        ['Plumas','California','CA'],
        ['Sonoma','California','CA'],
        ['Marin','California','CA'],
        ['Alameda','California','CA']]),
        columns=['name','enclosed_by','code']) 
    gg = update_list
    save_path = cv.graphics_path
    cv.graphics_path = cv.cv_home+'PlotsToShare/'
    plt.rc('figure', max_open_warning = 0)
    for g in range(0,len(gg)):
        print(gg['name'][g])
        tmpG = Geography(name=gg['name'][g], enclosed_by=gg['enclosed_by'][g],
                         code=gg['code'][g])
        tmpG.read_nyt_data('county')
        tmpG.plot_prevalence(signature = True, save=True)

    cv.graphics_path = save_path

def plot_multi_per_capita(mult = 1000,plot_dt=False,save=False):
    gg = pd.read_csv(cv.cv_home+'top30.csv',header=0,comment='#')
    print(gg.columns)
#   print(gg)

    key_cols = ('key','name','code')
    key = pd.DataFrame(columns=key_cols)

    fig, ax = plt.subplots(1,figsize=(6.5,4.5))
    firstDate = mdates.date2num(cv.FirstNYTDate)
    orderDate = mdates.date2num(cv.CAOrderDate)
    lastDate  = mdates.date2num(cv.EndOfTime)
    if (plot_dt):
        ax.set_ylabel('New Cases'+' per '+str(mult))
    else:
        ax.set_ylabel('Cases'+' per '+str(mult))
    ax.set_xlim([firstDate,lastDate])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(plt.MultipleLocator(30))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

    for g in range(0,len(gg)):
        print(gg['name'][g])
        tmpG = Geography(name=gg['name'][g], enclosed_by=gg['enclosed_by'][g],
                         code=gg['code'][g])
        tmpG.read_nyt_data('county')
        cases =  mult*tmpG.cases/tmpG.population + eps
        delta_cases = np.diff(cases)
        Date = tmpG.get_pdate()
        if (plot_dt):
            ax.plot(Date[1:], delta_cases, linewidth=1)#,alpha=0.75)
        else:
            ax.plot(Date, cases, linewidth=2)#,alpha=0.75)

        sn = short_name(tmpG.moniker)
    #   print(tmpG.name,tmpG.moniker,sn)
        kr = pd.Series((sn,tmpG.name,tmpG.code),index=key_cols)
        key = key.append(kr,ignore_index=True)

        if (plot_dt):
            mark_ends(ax,Date,delta_cases,sn,'r')
        else:
            mark_ends(ax,Date,cases,sn,'r')

#   Newsome's shelter in place order
    ax.plot((orderDate,orderDate),
    #       (ax.get_ylim()[0], ax.get_ylim()[1]),color='black',
            (0, ax.get_ylim()[1]),color='0.5',
            linewidth=3,alpha=0.5)

#   print(key)
    if save:
        gfile = cv.graphics_path+'counties_per_capita.eps'
        plt.savefig(gfile) #,dpi=300)
        plt.show(False)
        print('plot saved as',gfile)

        kfile = cv.graphics_path+'short_name_key.csv'
        key.sort_values(by='key',ascending=True,inplace=True)
        print(key)
        key.to_csv(kfile,index=False)
        print('key names saved as',kfile)

        plt.pause(3)
        plt.close()
            
    else:
        plt.show()


def update_everything():
    web_update()
    print('Finished web_update ...')
    os.system('rm -v '+ cv.dat_path + '*.dat')
    make_dat_files()
    plot_multi_per_capita(plot_dt=False,save=True)
    print('Finished make_dat_files()')
    update_shared_plots()
    print('Finished update_shared_plots()')
    os.system('rm -v' + cv.fit_path + '*.RData')
    update_fits()
    print('Finished update_fits()')
    make_fit_plots()
    print('Finished fit_plots')
    make_fit_table()
    make_fit_table()
    make_rate_plots('logbeta',add_doubling_time = True,save=True)
    make_rate_plots('logbeta',add_doubling_time = True,save=True,
                    fit_files=['NassauNY','Miami-DadeFL','HonoluluHI'])
    make_rate_plots('logmu',save=True)
    print('Finished rate_plots')
    print('Finished Everything!')


# --------------------------------------------------       
print('------- here ------')

#alam = Geography(name='Alameda',enclosed_by='California',code='CA')
#alam.read_nyt_data('county')
#alam.plot_prevalence(save=False,signature=True)
#alam.get_pdate()
#alam.print_metadata()
#alam.print_data()

#tfit = Fit(cv.fit_path+'unconstrained/'+'NassauNY.RData') #'Los Angeles','California','CA','ADMB')
#tfit = Fit(cv.fit_path+'unconstrained/'+'Miami-DadeFL.RData') #'Los Angeles','California','CA','ADMB')
#tfit.print_metadata()
#tfit.plot(save=True,logscale=False)


#update_everything()
#web_update()
#update_shared_plots()
#make_dat_files()
#update_fits()
#plot_multi_per_capita(plot_dt=False,save=True)
#make_fit_plots()
#make_fit_table()
cv.fit_path = cv.fit_path+'constrainID/'
make_rate_plots('logbeta',add_doubling_time = True,save=True)
make_rate_plots('logbeta',add_doubling_time = True,save=False,fit_files=['NassauNY','CookIL','Miami-DadeFL','HonoluluHI'])
make_rate_plots('logmu',save=True)
make_rate_plots('gamma',save=True)

#test = Geography(name='Nassau',enclosed_by='New York',code='NY')
#test = Geography(name='Miami-Dade',enclosed_by='Florida',code='FL')
#test = Geography(name='New York City',enclosed_by='New York',code='NY')
#test.read_nyt_data()
#test.write_dat_file()
#test.print_metadata()
#test.plot_per_capita_curvature()
#test.plot_prevalence(per_capita=True,save=False)#yscale='log',plot_dt=True)
#make_rate_plots('logbeta',add_doubling_time = True,save=True,fit_files=['NassauNY','Miami-DadeFL','HonoluluHI'])
#make_rate_plots('logbeta',add_doubling_time = True,save=True)
#make_rate_plots('logmu',save=True)
#make_rate_plots('gamma',save=True)

#plot_dow_boxes()
#plot_multi_per_capita(plot_dt=False,save=True)
#get_mu_atD1()

