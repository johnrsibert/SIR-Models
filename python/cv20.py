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

def mark_ends(ax,x,y,label,end='b',spacer=' '):
    c = ax.get_lines()[-1].get_color()
    if ( (end =='l') | (end == 'b')):
        ax.text(x,y,label+spacer,ha='right',va='center',fontsize=8,
                color=c)

    if ( (end =='r') | (end == 'b')):
        ax.text(x,y,spacer+label,ha='left',va='center',fontsize=8,
                color=c)

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
        O.write('# updateed from https://github.com/nytimes/covid-19-data.git\n')
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
                        window=[5,14], plot_dt = False, save = True):
        """ 
        Plots cases and deaths vs calendar date 
        """
        mult = 1000
    
        firstDate = mdates.date2num(cv.FirstNYTDate)
        orderDate = mdates.date2num(cv.CAOrderDate)
        lastDate  = mdates.date2num(cv.EndOfTime)
    
        date_list = pd.date_range(start=firstDate,end=lastDate)
    #    date_lim = [date_list[0],date_list[len(date_list)-1]]
    
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
        
        if (per_capita):
            cases =  mult*self.cases/self.population + eps
            deaths =  mult*self.deaths/self.population + eps
        else :
            cases  =  self.cases
            deaths =  self.deaths
        
        Date = self.get_pdate()
        
        c = ax[0].plot(Date, cases)#,label=cc)
        tcol = ax[0].get_lines()[0].get_color()
        mark_ends(ax[0],Date[len(Date)-1],cases[len(cases)-1],'C','r')
        if (delta_ts):
            delta_cases = np.diff(cases)
            ax2[0].bar(Date[1:], delta_cases, alpha=0.5)
            for w in range(0,len(window)):
                adc = pd.Series(delta_cases).rolling(window=window[w]).mean()
                ax2[0].plot(Date[1:],adc,linewidth=1)
                mark_ends(ax2[0],Date[len(Date)-1],adc[len(adc)-1],
                          str(window[w])+'da','r')

        if ((yscale == 'log') & (plot_dt)):
            ax[0] = self.plot_dtslopes(ax[0])
                
        d = ax[1].plot(Date, deaths)#,label=cc)
        mark_ends(ax[1],Date[len(Date)-1],deaths[len(deaths)-1],'D','r')
        if (delta_ts):
            delta_deaths = np.diff(deaths)
            ax2[1].bar(Date[1:],delta_deaths,alpha=0.5)
            for w in range(0,len(window)):
                add = pd.Series(delta_deaths).rolling(window=window[w]).mean()
                ax2[1].plot(Date[1:],add,linewidth=1)
                mark_ends(ax2[1],Date[len(Date)-1],add[len(add)-1],
                          str(window[w])+'da','r')
    
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
    
        title = 'COVID-19 Prevalence in '+self.name+' County, '+self.enclosed_by
    #   fig.text(0.5,0.925,title ,ha='center',va='top')
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
    
        if save:
            plt.savefig(cv.graphics_path+self.moniker+'_prevalence.png',dpi=300)
        else:
            plt.show()


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
            mark_ends(ax,xrange[1],y,str(dt[i])+' da','r')

        return(ax)


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
            prefix = 'log '
             
        fig, ax = plt.subplots(npl,1,figsize=(10.0,(npl)*2.5))
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
            ax[a].set_xlim([firstDate,lastDate])
            ax[a].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax[a].xaxis.set_major_locator(plt.MultipleLocator(30))
            ax[a].xaxis.set_minor_locator(plt.MultipleLocator(1))
    #       if (delta_ts):
    #           ax2.append(ax[a].twinx())
    #   ax[1].xaxis.set_minor_locator(plt.MultipleLocator(1))
    
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
        sigma_beta = np.exp(self.get_est_or_init('logsigma_beta'))
        sigma_mu = np.exp(self.get_est_or_init('logsigma_mu'))
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
           ymin = min(self.diag['beta'])-2.5*sigma_beta
           ax[2].set_ylim(ymin,1.2*max(self.diag['beta']))
           sigstr = '%s = %.3g'%('$\sigma_\\beta$',sigma_beta)
           tx = prop_scale(ax[2].get_xlim(), 0.05)
           ty = prop_scale(ax[2].get_ylim(), 0.90)
           ax[2].text(tx,ty,sigstr, ha='left',va='center',fontsize=10)
           ax[2].plot(ax[2].get_xlim(),[0.0,0.0],color='0.2',linestyle='--')
           ax[2].plot(pdate,self.diag['beta'])
           plot_error(ax[2],pdate,self.diag['beta'],sigma_beta)
           med = median(self.diag['beta'])
           medstr = '%s = %.3g'%('$\\tilde{\\beta}$',med)
           ax[2].text(ax[2].get_xlim()[0],med,medstr,ha='left',va='bottom',fontsize=10)
           ax[2].plot(ax[2].get_xlim(),[med,med])

           y2_ticks = np.log(2)/(ax[2].get_yticks()+eps)
           n = len(y2_ticks)
           y2_tick_label = ['']*n
           for i in range(0,len(y2_ticks)):
              if (y2_ticks[i] > 7):
                  y2_tick_label[i] = ' >14'
              else:
                 y2_tick_label[i] = '%.1f'%y2_ticks[i]
           ax[2].grid(False,axis='y')
           dtax = ax[2].twinx()
           dtax.set_yscale(ax[2].get_yscale())
           dtax.set_ylabel('Doubling Time (da)')
           dtax.set_yticks(ax[2].get_yticks())
           dtax.set_yticklabels(y2_tick_label)
           dtax.set_ylim(ax[2].get_ylim())

        if (npl > 3):
           ymin = min(self.diag['mu'])-2.5*sigma_mu
           ax[3].set_ylim(ymin,1.2*max(self.diag['mu']))
           sigstr = '%s = %.3g'%('$\sigma_\\mu$',sigma_mu)
           tx = prop_scale(ax[3].get_xlim(), 0.05)
           ty = prop_scale(ax[3].get_ylim(), 0.90)
           ax[3].plot(ax[2].get_xlim(),[0.0,0.0],color='0.2',linestyle='--')
           ax[3].plot(pdate,self.diag['mu'])
           plot_error(ax[3],pdate,self.diag['mu'],sigma_mu)
           ax[3].text(tx,ty,sigstr, ha='left',va='center',fontsize=10)
           med = median(self.diag['mu'])
           medstr = '%s = %.3g'%('$\\tilde{\\mu}$',med)
           ax[3].text(ax[3].get_xlim()[0],med,medstr,ha='left',va='bottom',fontsize=10)
           ax[3].plot(ax[3].get_xlim(),[med,med])
    
    #   title = self.name+' County, '+self.enclosed_by
    #   fig.text(0.5,0.925,title ,ha='center',va='top')

        if save:
            gfile = cv.graphics_path+self.moniker+'_'+self.fit_type+'_estimates'
            plt.savefig(gfile+'.png',dpi=300)
            plt.show(False)
            print('plot saved as',gfile)
        else:
            plt.show(True)



# ----------- end of class definitions--------------       

def make_fit_plots(ext = '.RData'):
    fit_files = glob.glob(cv.fit_path+'*'+ext)
    print('found',len(fit_files),ext,'files in',cv.fit_path)
    plt.rc('figure', max_open_warning = 0)
    for ff in fit_files:
        fit = Fit(ff)
    #   fit.print_metadata()
        fit.plot(logscale=False)



def make_fit_table(ext = '.RData'):
    fit_files = glob.glob(cv.fit_path+'*'+ext)
    print('found',len(fit_files),ext,'files in',cv.fit_path)

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
    #   fn = ff.replace(' ','_',5) 
    #   pn = fit_path+fn+'.RData'
    #   fit=pyreadr.read_r(pn)
        print('adding fit',ff)
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
        tmp = fit.get_est_or_init('loggamma')
        tmp = np.exp(tmp)
        tmp = float(tmp)
        gamm = np.append(gamm,tmp)
        func = np.append(func,float(fit.get_metadata_item('fn')))
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

    tex = cv.fit_path+'fit_table.tex'
    ff = open(tex, 'w')
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
    gg = pd.read_csv(cv.cv_home+'UpdateList.csv',header=0,comment='#')
    print(gg.columns)
#   print(gg)
    plt.rc('figure', max_open_warning = 0)
    for g in range(0,len(gg)):
        print(gg['name'][g])
        tmpG = Geography(name=gg['name'][g], enclosed_by=gg['enclosed_by'][g],
                         code=gg['code'][g])
        tmpG.read_nyt_data('county')
        tmpG.write_dat_file()
        tmpG.plot_prevalence(save=True)

def update_fits():
    save_wd = os.getcwd()
    print('save:',save_wd)
    print(cv.TMB_path)
    os.chdir(cv.TMB_path)
    print('current',os.getcwd())
    cmd = 'R CMD BATCH runSS4.R'
    print('running',cmd)
    os.system(cmd)
    os.chdir(save_wd)
    print('current',os.getcwd())


# cv_home = '/home/jsibert/Projects/SIR-Models/'

#   make_dat_file()
#   plot_county_dat(county_dat,County='Alameda',State='California',file='AlamedaCA')
#   plot_county_dat(county_dat,County='Marin',State='California',file='MarinCA')
#   plot_county_dat(county_dat,County='Sonoma',State='California',file='SonomaCA')
#   plot_county_dat(county_dat,County='Honolulu',State='Hawaii',file='HonoluluHI')
#   plot_county_dat(county_dat,County='Tompkins',State='New York',file='TompkinsNY')
#   plot_county_dat(county_dat,County='Placer',State='California',file='PlacerCA')



# --------------------------------------------------       
#alam = Geography(name='Alameda',enclosed_by='California',code='CA')
#alam.read_nyt_data('county')
#alam.get_pdate()
#alam.print_metadata()
#alam.get_pdate()
#alam.print_data()
#alam.plot_prevalence()
print('------- here ------')
#tfit = Fit(cv.fit_path+'Los_AngelesCA.RData') #'Los Angeles','California','CA','ADMB')
#tfit.print_metadata()
#tfit.plot()

#make_fit_table()
#make_fit_table('.rep')

update_fits()
make_fit_plots()

#test = Geography(name='District of Columbia',enclosed_by='District of Columbia',code='DC')
#test.read_nyt_data()
#test.write_dat_file()
#test.print_metadata()
#test.plot_prevalence()#yscale='log',window=[7,14],per_capita=True) #,plot_dt=True)

#plot_dow_boxes()

#web_update()
#make_dat_files()
