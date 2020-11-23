#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taking a more OO approach to the code in cv19.py

Created on Thu Jul  2 09:04:11 2020

@author: jsibert
"""

import pandas as pd
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc
import numpy as np
import os
import sys
import pyreadr
from io import StringIO 
from io import BytesIO
import base64
import scipy.stats as stats
from sigfig import round
from tabulate import tabulate
from collections import OrderedDict
import glob
import re
import statistics

plt.style.use('file:///home/jsibert/.config/matplotlib/john.mplstyle')
import js_covid as cv

eps = 1e-5

# ---------------- global utility functions ---------------------------

def make_nyt_census_dat():
    """
    Generate file with census population estimates 
    and county names  updated from https://github.com/nytimes/covid-19-data.git
    add two byte state postal codes
    add 'flag' field for selecting favorites
    add population estimate for New York City as per NYT practice
    """
    census_dat_file = cv.cv_home+'co-est2019-pop.csv'
    census_dat = pd.read_csv(census_dat_file,header=0,comment='#')
    census_dat = census_dat[census_dat['COUNTY']>0]
    census_dat['fips'] = 0
#   generate fips from concatenation of STATE and COUNTY fields in census records
    for r in range(0,len(census_dat)):
        fips = '{:02d}{:03d}'.format(census_dat['STATE'].iloc[r],census_dat['COUNTY'].iloc[r])
    #   if (census_dat['county'].iloc[r] == 'Kalawao'):
    #       print('----------Kalawao---------',census_dat['population'].iloc[r])
    #       print(r,census_dat['COUNTY'].iloc[r],census_dat['STATE'].iloc[r],
    #               census_dat['county'].iloc[r],census_dat['state'].iloc[r], fips)
        census_dat['fips'].iloc[r] = int(fips)

#   aggregate populations of NYC borroughs into NY Times convention for 
    nyc_counties = ('Queens','Richmond','Kings','Bronx','New York')
    nyc_c_filter = (  census_dat['county'].isin(nyc_counties) 
                    & census_dat['state'].isin(['New York']))
    nyc_population = int(census_dat[nyc_c_filter]['population'].sum())

#   remove nyc_counties from census data
    census_dat = census_dat[~nyc_c_filter] 

#   create unique instances of fips,population combinations using set(..)
    county_state_pop = set(zip(census_dat['fips'],census_dat['population']))
    cs_pop = pd.DataFrame(county_state_pop,columns=('fips','population'))

#   get NYT data    
    nyt_dat = pd.read_csv(cv.NYT_counties,header=0)
    nyt_dat = nyt_dat.sort_values(by=['fips'],ascending=False)
#   remove counties without fips code (mainly 'Unknown' counties
    empty_fips_filter = pd.notna(nyt_dat['fips'])
    nyt_dat = nyt_dat[empty_fips_filter]

#   create unique instances of NYT county & state combinations
    county_state_nyt = set(zip(nyt_dat['county'],nyt_dat['state'],nyt_dat['fips']))
    nyt_counties = pd.DataFrame(county_state_nyt,columns=('county','state','fips'))
    nyt_counties['code'] = None
    nyt_counties['flag'] = ' '
    nyt_counties = nyt_counties.sort_values(by=['fips'],ascending=False)

#   insert state postal codes and other abbreviation
    gcodes = pd.read_csv(cv.cv_home+'geography_codes.csv',header=0,comment='#')
    for i in range(0,len(gcodes)):
        geog = gcodes['geography'].iloc[i]
        code = gcodes['code'].iloc[i]
        gfilter = nyt_counties['state'].isin([geog])
    #   avoid 'SettingWithCopyWarning': 
        nyt_counties.loc[gfilter,'code'] = code
   
#   merge the two data frames using NYT county designations
    nyt_census = nyt_counties.merge(right=cs_pop)
#   append row for New York City population
    nyc_row = pd.Series(['New York City','New York',36999,'NY',' ',nyc_population],index=nyt_census.columns)
    nyt_census = nyt_census.append(nyc_row,ignore_index=True)
#   df.astype({'col1': 'int32'}).dtype
    nyt_census = nyt_census.astype({'fips':'int64'})
    nyt_census = nyt_census.sort_values(by=['population','state','county'],ascending=False)

    print('nyt_census (Los Angeles, CA through Kenedy, TX):')
    print(nyt_census)
    nyt_census.to_csv('nyt_census.csv',index=False)

def Strptime(x):
    """
    wrapper for datetime.strptime callable by map(..)
    """
    y = datetime.strptime(x,'%Y-%m-%d')
    return(y)

def prop_scale(lim,prop):
    s = lim[0] + prop*(lim[1]-lim[0])
    return(s)

def SD_lim(x, mult):
    M = statistics.mean(x)
    S = statistics.stdev(x)
    if (S > 0.0):
        multS = mult*S
        return([M-multS,M+multS])
    else:
        return(min(x),max(x))
    
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

def none_reported(ax,what):
    lim = ax.get_xlim()
    tx = lim[0] + 0.5*(lim[1]-lim[0])
    lim = ax.get_ylim()
    ty = 0.5*lim[1] #lim[0] + 0.5*(lim[1]-lim[0])
    note = 'No '+what+' Reported'
    print(note)
    ax.text(tx,ty,note,ha='center',va='center',fontstyle='italic')


def mark_peak(ax,x,y,label):
    c = ax.get_lines()[-1].get_color()
    a = ax.get_lines()[-1].get_alpha()
    i = pd.Series(y).idxmax()
    if (i>0):
      mark = ax.text(x[i],y[i],label,ha='center',va='bottom',fontsize=8, 
                     color=c) #,alpha=a)
      mark.set_alpha(a) # not supported on all backends

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
    sd_region = plt.Polygon(np.transpose(xy), alpha=0.2,
                            facecolor=c, edgecolor='0.1',lw=0.5)
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
            self.moniker = str(self.name+self.code)
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
        print('gntime:',self.ntime)

    def print_data(self):
        self.get_pdate()
    #   if (self.deaths != None):
        if (len(self.deaths) > 0):
            print('date','cases','deaths','pdate')
        else:
            print('date','cases','pdate')
        for k in range(0,len(self.date)):
        #   if (self.deaths != None):
            if (len(self.deaths) > 0):
                print(self.date[k],self.cases[k],self.deaths[k],self.pdate[k])
            else:
                print(self.date[k],self.cases[k],self.pdate[k])
            
    def to_DataFrame(self):
        self.get_pdate()
        Gdf = pd.DataFrame(columns=('date','pdate','cases','deaths'))
        Gdf['date'] = self.date
        Gdf['pdate'] = self.pdate
        Gdf['cases'] = self.cases
        Gdf['deaths'] = self.deaths
        return(Gdf)

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
        
        """

        dat = cv.population_dat
        if (dat.empty):
            print('Reading',cv.census_data_path)
            cv.population_dat = pd.read_csv(cv.census_data_path,header=0,comment='#')
            dat = cv.population_dat
    #   else:
    #       print('Using current "dat" object')

        state_filter = dat['state'].isin([self.enclosed_by])
        county_filter = dat['county'].isin([self.name])
    #   COUNTY_filter = (dat['COUNTY']>0)
        County_rows = state_filter & county_filter #& COUNTY_filter
        try:
            population = int(pd.to_numeric(dat[County_rows]['population'].values))
        except:
            print('get_county_pop() failed for:')
            print(self.name, self.enclosed_by,self.code) 
            population = 1
        return(population)   
        
    def read_nyt_data(self,gtype='county'):
        self.gtype = gtype
        mtime = 0.0
        if (gtype == 'county'):
            dat = cv.nyt_county_dat
            if (dat.empty):
               cspath = cv.NYT_counties
               print('Reading',cspath)
               cv.nyt_county_dat = pd.read_csv(cspath,header=0)
               dat = cv.nyt_county_dat
               mtime = os.path.getmtime(cspath)
        else:
            sys.exit('No data for',gtype)
        
    #   dat = pd.read_csv(cspath,header=0)

    #   mtime = os.path.getmtime(cspath)
        dtime = datetime.fromtimestamp(mtime)
        self.updated = str(dtime.date())
        self.population = self.get_county_pop()
    
        state_filter = dat['state'].isin([self.enclosed_by])
        county_filter = dat['county'].isin([self.name])
        County_rows = state_filter & county_filter
        if (len(County_rows) < 1):
            sys.exit(' * * * no records found for',self.name,self.surrounded_by)

   #    tmp = dat[County_rows]['date'].map(Strptime)
   #    self.pdate  = np.array(mdates.date2num(tmp))
        self.cases  = np.array(dat[County_rows]['cases'])
        self.deaths = np.array(dat[County_rows]['deaths'])
        self.date   = np.array(dat[County_rows]['date'])
        self.date0 = self.date[0]
        self.ntime = len(self.date)
        self.source = 'New York Times, https://github.com/nytimes/covid-19-data.git.'
        
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

    def read_BCHA_data(self,gtype='hsa'):
        self.gtype = gtype
        cspath = cv.BCHA_path

        dat = pd.read_csv(cspath,header=0)

        mtime = os.path.getmtime(cspath)
        dtime = datetime.fromtimestamp(mtime)
        self.updated = str(dtime.date())
        self.population = None
   #    self.source = 'British Columbia Centre for Disease Control'\
        self.source = 'Government of British Columbia'\
                      ' www.bccdc.ca/Health-Info-Site/Documents/' 

        rdates = pd.DatetimeIndex(dat["Reported_Date"],normalize=True)
        self.date = pd.date_range(rdates[0],rdates[len(rdates)-1],normalize=True,freq='D')
        self.date = self.date.strftime('%Y-%m-%d') # NYT style
        self.date0 = self.date[0]
        self.ntime = len(self.date)
        self.get_pdate()


        daily_cases = pd.Series([0]*self.ntime)

        HA_filter = dat["HA"].isin([self.name])
        dat["rep_date"] = rdates
        for i in range(0,len(self.date)):
        #   date_mask = dat["Reported_Date"].isin([self.date[i]])
            date_mask = dat["rep_date"].isin([self.date[i]])
            mask = HA_filter & date_mask
            daily_cases[i] = len(dat[mask.values])

        self.cases = daily_cases.cumsum()
        


    def plot_prevalence(self,yscale='linear', per_capita=False, delta_ts=True,
                        window=[11], plot_dt = False, cumulative = True,
                        show_order_date = True,
                        annotation = True, signature = False, 
                        save = True, dashboard = False):
        
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

        firstDate = datetime.strptime(self.date[0],'%Y-%m-%d')
        orderDate = mdates.date2num(cv.CAOrderDate)

        if (self.deaths is None):
            nax = 1
        else:
            nax = 2
    
        fig, pax = plt.subplots(nax,1,figsize=(6.5,nax*2.25))
        ax = [None]*2
        if (nax == 1):
            ax[0] = pax
        else:
            ax=pax

        if (per_capita):
            ax[0].set_ylabel('Daily Cases'+' per '+str(mult))
            if (nax > 1):
                ax[1].set_ylabel('Daily Deaths'+' per '+str(mult))
        else:
            ax[0].set_ylabel('Daily Cases')
            if (nax > 1):
                ax[1].set_ylabel('Daily Deaths')
    
        ax2 = []
        for a in range(0,nax):
            self.make_date_axis(ax[a],firstDate)
            ax[a].set_yscale(yscale)
            if (cumulative):
                ax2.append(ax[a].twinx())
                ax2[a].set_ylabel('Cumulative')
        
        if (per_capita):
            cases =  mult*self.cases/self.population + eps
            if (nax > 1):
                deaths =  mult*self.deaths/self.population + eps
            else:
                deaths =  self.deaths
        else:
            cases  =  self.cases
            deaths =  self.deaths
        
        nn = self.ntime-1
        max_cases = cases[nn]
        if (self.deaths is None):
            max_deaths = 0.0
        else:
            max_deaths = deaths[nn]

        Date = self.get_pdate()

        delta_cases = np.diff(cases)
        ax[0].bar(Date[1:], delta_cases)

        if (max_cases < 1.0):
            none_reported(ax[0],'Cases')
        else:
            for w in range(0,len(window)):
                adc = pd.Series(delta_cases).rolling(window=window[w]).mean()
                ax[0].plot(Date[1:],adc,linewidth=2)
                mark_ends(ax[0],Date[1:],adc, str(window[w])+'da','r')
        
            if (cumulative):
                ax2[0].plot(Date, cases,alpha=0.5, linewidth=1)#,label=cc)
                mark_ends(ax2[0],Date,cases,r'$\Sigma$C','r')

            if ((yscale == 'log') & (plot_dt)):
                ax[0] = self.plot_dtslopes(ax[0])
                
        if (nax > 1):
            if (cumulative):
                ax2[1].plot(Date, deaths,alpha=0.5,linewidth=1)#,label=cc)
                mark_ends(ax2[1],Date,deaths,r'$\Sigma$D','r')

            if (max_deaths > 0.0):
                delta_deaths = np.diff(deaths)
                ax[1].bar(Date[1:],delta_deaths)
                for w in range(0,len(window)):
                    add = pd.Series(delta_deaths).rolling(window=window[w]).mean()
                    ax[1].plot(Date[1:],add,linewidth=2)
                    mark_ends(ax[1],Date[1:],add, str(window[w])+'da','r')
            else:
                none_reported(ax[1],'Deaths')
    
    #   Adjust length of y axis
        ax[0].set_ylim(0.0,SD_lim(delta_cases,3.0)[1]) #ax[a].get_ylim()[1])
        if (max_deaths > 0.0):
            ax[1].set_ylim(0.0,SD_lim(delta_deaths,3.0)[1]) #ax[a].get_ylim()[1])

        for a in range(0,nax):
            if (delta_ts):
                if (cumulative):
                    ax2[a].set_ylim(0,ax2[a].get_ylim()[1])
        #   Newsome's shelter in place order
            if (show_order_date):
                ax[a].plot((orderDate,orderDate),
                           (ax[a].get_ylim()[0], ax[a].get_ylim()[1]),
                           color='black', linewidth=3,alpha=0.5)
        #   ax[a].legend()
    
        if (annotation):
            if (self.gtype == 'county'):
                gname = 'County'
            else:
                gname = 'Region'
            title = 'Covid-19 Prevalence in '+self.name+' '+gname+', '+ self.enclosed_by
            fig.text(0.5,1.0,title ,ha='center',va='top')
            fig.text(0.0,0.0,' Data source: '+ self.source ,
                     ha='left',va='bottom', fontsize=8)
    
            mtime = os.path.getmtime(cv.NYT_home+'us-counties.csv')
            dtime = datetime.fromtimestamp(mtime)
            fig.text(1.0,0.0,'Updated '+str(dtime.date())+' ', ha='right',va='bottom', fontsize=8)

        if (signature):
            by_line = 'Graphics by John Sibert'
            url_line = 'https://github.com/johnrsibert/SIR-Models/tree/master/PlotsToShare'
            fig.text(0.0,0.025,' '+by_line, ha='left',va='bottom', fontsize=8,alpha=0.25)#,color='red')
            fig.text(1.0,0.025,url_line+' ', ha='right',va='bottom', fontsize=8,alpha=0.25)#,color='red')
    
        if (dashboard):
        #   out_img = BytesIO()
        #   plt.savefig(out_img, format='png')
        #   out_img.seek(0)  # rewind file
        #   encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
        #   return "data:image/png;base64,{}".format(encoded)
            gfile = cv.graphics_path+'test.png'
            
            print('saving fig for dash:',gfile)
            plt.savefig(gfile,format='png',dpi=300)
            plt.close()
            print('saved')

        else:
            if save:
                gfile = cv.graphics_path+self.moniker+'_prevalence.png'
                plt.savefig(gfile,dpi=300)
                plt.show(False)
                plt.pause(3)
                plt.close()
                print('plot saved as',gfile)
            else:
                plt.show()


    def make_date_axis(self, ax, first_prev_date = None):
        if (first_prev_date is None):
            firstDate = mdates.date2num(cv.FirstNYTDate)
        else:
            firstDate = first_prev_date
        
        lastDate  = mdates.date2num(cv.EndOfTime)
    #   print('firstDate,lastDate:',firstDate,lastDate)
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
        if (cv.pyreadr_kludge):
        #   print('keys',tfit.keys())
        #   print(tfit['diag'].columns)
        #   print(len(tfit['diag'].columns))
            insert = len(tfit['diag'].columns)
        #   print('head:',head,'tail:',tail,tail[0])
        #   print('filename',filename,'extension',extension)
            csv_file = filename + '_stderror.csv'
        #   print('reading',csv_file)
            stderror = pd.read_csv(csv_file,header=0)
        #   print(stderror['logbeta'],stderror['logmu'])
        #   SElogbeta = stderror['logbeta']
        #   SElogmu = stderror['logmu']
            self.diag.insert(insert,'SElogbeta',stderror['logbeta'])
            self.diag.insert(insert+1,'SElogmu',stderror['logmu'])
        #   print(self.diag)

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
            ax[2].set_ylabel(r'$\ln\ \beta\ (da^{-1})$')
        if (npl > 3):
            ax[3].set_ylabel(r'$\ln\ \mu\ (da^{-1})$')
    
    
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
        SElogbeta = self.diag['SElogbeta']
        SElogmu   = self.diag['SElogmu']
        sigma_logC = np.exp(self.get_est_or_init('logsigma_logC'))
        sigma_logD = np.exp(self.get_est_or_init('logsigma_logD'))
        sigma_logbeta = self.get_est_or_init('logsigma_logbeta')
        sigma_logmu = np.exp(self.get_est_or_init('logsigma_logmu'))
        if (logscale):
            ax[0].set_ylim(0.0,1.2*max(obsI))
            ax[0].scatter(pdate,obsI,marker = '.',s=16,c='r')
            ax[0].plot(pdate,preI)

        else:
            ax[0].set_ylim(0.0,1.2*max(np.exp(obsI)))
            ax[0].scatter(pdate,np.exp(obsI),marker = '.',s=16,c='r')
            ax[0].plot(pdate,np.exp(preI))
        plot_error(ax[0],pdate,obsI,sigma_logC,logscale)
        tx = prop_scale(ax[0].get_xlim(), 0.05)
        ty = prop_scale(ax[0].get_ylim(), 0.90)
        sigstr = '%s = %.3g'%('$\sigma_I$',sigma_logC)
        ax[0].text(tx,ty,sigstr, ha='left',va='center',fontsize=10)
    
        if (logscale):
            ax[1].set_ylim(0.0,1.2*max(obsD))
            ax[1].scatter(pdate,obsD,marker = '.',s=16,c='r')
            ax[1].plot(pdate,preD)
        else:
            ax[1].set_ylim(0.0,1.2*max(np.exp(obsD)))
            ax[1].scatter(pdate,np.exp(obsD),marker = '.',s=16, c='r')
            ax[1].plot(pdate,np.exp(preD))

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
        #   plot_error(ax[2],pdate,log_beta,sigma_logbeta,logscale=True)
            plot_error(ax[2],pdate,log_beta,SElogbeta,logscale=True)
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
            #   fig.show()
            #   fig.canvas.draw()
                y2_ticks = dtax.get_yticks()
                labels = dtax.get_yticklabels()
                labels[0] = ''
                for i in range(1,len(y2_ticks)):
                    y2_ticks[i] = np.log(2)/np.exp(y2_ticks[i])
                    labels[i] = round(float(y2_ticks[i]),2)
                      
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
        #   plot_error(ax[3],pdate,logmu,sigma_logmu,logscale=True)
            plot_error(ax[3],pdate,logmu,SElogmu,logscale=True)
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
                gfile = cv.graphics_path+self.moniker+'_'+self.fit_type+'_estimates_a'
            fig.savefig(gfile+'.pdf')#,dpi=300)
            plt.show(False)
            print('plot saved as',gfile)
            plt.pause(3)
            plt.close()
        else:
            plt.show(True)

    def plot_CMR(self, save = True):
        npl = 3
        fig, ax = plt.subplots(npl,1,figsize=(9.0,(npl)*3.0))
        ax[0].set_ylabel('Empirical CMR')
        ax[1].set_ylabel('Estimated CMR')
        ax[2].set_ylabel(r'$\mu\ (da^{-1})$')
 
        for a in range(0,len(ax)):
            self.make_date_axis(ax[a])
    
        Date0 = self.get_metadata_item('Date0')
        Date0 = datetime.strptime(Date0,'%Y-%m-%d')
        pdate = []
        for t in range(0,len(self.diag.index)):
            pdate.append(mdates.date2num(Date0 + timedelta(days=t)))
        
    #   self.get_pdate()
        obsI = self.diag['obs_cases']
        obsD = self.diag['obs_deaths']
        obsCMR = obsD/obsI
        estI = np.exp(self.diag['log_pred_cases'])
        estD = np.exp(self.diag['log_pred_deaths'])
        estCMR = estD/estI 
        mu   = np.exp(self.diag['logmu'])

        ax[0].plot(pdate,obsCMR)
        ax[1].set_ylim(ax[0].get_ylim())
        ax[1].plot(pdate,estCMR)
        ax[2].plot(pdate,mu)

        fig.text(0.5,0.95,'Case Mortality Ratio' ,ha='center',va='bottom')
        if (save):
            gfile = cv.graphics_path+self.moniker+'_CMR.pdf'
            fig.savefig(gfile)
            plt.show(False)
            print('plot saved as',gfile)
            plt.pause(3)
        else:
            plt.show()

# ----------- end of class definitions--------------       

def make_SD_tab(Gfile='top30.csv',save=True):

    def robust_sd(a):
        mask = [~np.isnan(a) & ~np.isinf(a)]
        sd = statistics.stdev(a[mask])
        return(sd)

    print('Reading:',cv.cv_home+Gfile)
    gg = pd.read_csv(cv.cv_home+Gfile,header=0,comment='#')#, encoding = "ISO-8859-3")
    print('Finished reading:',cv.cv_home+Gfile)
    print(gg.columns)

    SD_columns=['County','Cases','Deaths','log Cases','log Deaths']
    SD_tab = pd.DataFrame(columns=SD_columns, dtype=None)
    row = pd.Series(index=SD_columns)

    print('Processing',len(gg),'geographies')
    for g in range(0,len(gg)):
        print(g,gg['name'][g])
        tmpG = Geography(name=gg['name'][g], enclosed_by=gg['enclosed_by'][g],
                         code=gg['code'][g])
        tmpG.read_nyt_data('county')

        dt_cases = np.diff(tmpG.cases)
        dt_deaths = np.diff(tmpG.deaths)
        dt_log_cases = np.diff(np.log(tmpG.cases))
        dt_log_deaths = np.diff(np.log(tmpG.deaths))

        row['County'] = tmpG.name
   #    row['County'] = pretty_county(tmpG.name)
        row['Cases'] = robust_sd(dt_cases)
        row['Deaths'] = robust_sd(dt_deaths)
        row['log Cases'] = robust_sd(dt_log_cases)
        row['log Deaths'] = robust_sd(dt_log_deaths)

        SD_tab = SD_tab.append(row,ignore_index=True)

#       print(type(dt_log_deaths),dt_log_deaths)
#       nan_filter = [~np.isnan(dt_log_deaths) & ~np.isinf(dt_log_deaths)]
#       print(type(nan_filter), nan_filter)
#       tt = pd.DataFrame([dt_log_deaths,nan_filter])
#       pd.set_option('display.max_rows', None)
#       print(tt.transpose())
#       print('sd',statistics.stdev(dt_log_deaths[nan_filter]))
#       print('robust',robust_sd(dt_log_deaths))

    print(SD_tab)    
    row['County'] = 'Median'
    row['Cases'] = median(SD_tab['Cases'])
    row['Deaths'] = median(SD_tab['Deaths'])
    row['log Cases'] = median(SD_tab['log Cases'])
    row['log Deaths'] = median(SD_tab['log Deaths'])
    SD_tab = SD_tab.append(row,ignore_index=True)
    print(SD_tab)    

#def plot_DC(Gfile='top30.csv',save=True):
def plot_DC(nG=30,save=True):

    def plot_cmr(a, rr=[2.0]):
        for r in rr:
            rstr = str(r)
            xr = [0.0]*2
            yr = [0.0]*2
            for i in range(0,len(yr)):
                xr[i] = a.get_xlim()[i]
                yr[i] = xr[i]*r/100.0

            a.plot(xr,yr, linewidth=1,color='0.1',alpha=0.5)  
            mark_ends(a,xr,yr,rstr,'r',' ')

    plt.rcParams["scatter.marker"] = '.'
    plt.rcParams["lines.markersize"] = 3
    nplot = 3
    fig, ax = plt.subplots(nplot,figsize=(6.5,9.0))

    for i in range(0,2):
        ax[i].set_yscale('log')
        ax[i].set_ylabel('Deaths')
        ax[i].set_ylim(1,1e5)
        ax[i].set_xscale('log')
        ax[i].set_xlabel('Cases')
        ax[i].set_xlim(10,1e6)


#   if (nplot > 2):
#      i = 2
#      ax[i].set_yscale('log')
#      ax[i].set_ylim(0.0025,0.16)
#      ax[i].set_ylabel('Fatality/Cases')
#      ax[i].set_xscale('log')
#      ax[i].set_xlabel('Cases')
#      ax[i].set_xlim(10,1e6)

    ct = []
    dt = []
    ft = []
    recent = pd.DataFrame(columns = ('moniker','cases','deaths','cfr'))
    gg = pd.read_csv(cv.census_data_path,header=0,comment='#')
    print('Processing',nG,'geographies')
    for g in range(0,nG):
        print(g,gg['county'].iloc[g])
        tmpG = Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                         code=gg['code'].iloc[g])
        tmpG.read_nyt_data('county')
        print(tmpG.moniker+'.RData')
     #  gdf = tmpG.to_DataFrame()
     #  gdf = gdf.sort_values(by='cases',ascending=False)
     #  print(gdf)
        
        ax[0].scatter(tmpG.cases,tmpG.deaths)
        nt = len(tmpG.cases)-1
        ct.append(tmpG.cases[nt])
        dt.append(tmpG.deaths[nt])
        cfrt =tmpG.deaths[nt]/tmpG.cases[nt]
        ft.append(cfrt)
        row = pd.Series(index=recent.columns)
        row['moniker'] = tmpG.moniker
        row['cases'] = tmpG.cases[nt]
        row['deaths'] = tmpG.deaths[nt]
        row['cfr'] = cfrt
        recent = recent.append(row, ignore_index=True)

        if (nplot > 2):
           ratio = tmpG.deaths/tmpG.cases
           ax[2].scatter(tmpG.cases,ratio,color='blue',alpha=0.2)

    recent = recent.sort_values(by='cases',ascending=False)
    recent.to_csv('recent_cfr.csv',index=False)
    print(recent)

    if (nplot > 2):
        ax[2].hist(ft,50)
        ax[2].set_xlim(0.0,0.1)
        ax[2].set_xlabel('Most Recent Case Fatality Ratio')
        ax[2].set_ylabel('Number')

    plot_cmr(ax[0], [0.5,1.0,2.0,4.0,8.0])
    logxlim = np.log(ax[0].get_xlim())
    tx = np.exp(logxlim[0]+0.05*(logxlim[1]-logxlim[0]))
    logylim = np.log(ax[0].get_ylim())
    ty = np.exp(logylim[0]+0.9*(logylim[1]-logylim[0]))
    ax[0].text(tx,ty,' n = '+str(nG),ha='left',va='center')

    ax[1].scatter(ct,dt)
    plot_cmr(ax[1], [0.5,1.0,2.0,4.0,8.0])
    ax[1].text(tx,ty,' n = '+str(nG)+', most recent',ha='left',va='center')

    if save:
        gfile = cv.graphics_path+'CFR_'+str(nG)+'.png'
        plt.savefig(gfile,dpi=300)
        plt.show(False)
        plt.pause(5)
            
        print('Plot saved as',gfile)
    else:
        plt.show()




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
        suffix = '_'+str(len(fit_files))
        for i,ff in enumerate(fit_files):
            fit_files[i] = cv.fit_path+fit_files[i]+ext
    for i,ff in enumerate(fit_files):
        print(i,ff)
        fit = Fit(ff)
        if (i == 0):
            fit.make_date_axis(ax)
            ax.set_ylabel(ylabel)
       
        pdate = []
        Date0 = datetime.strptime(fit.date0,'%Y-%m-%d')
        for t in range(0,len(fit.diag.index)):
            pdate.append(mdates.date2num(Date0 + timedelta(days=t)))

        yvar = fit.diag[yvarname]
        if (yvarname == 'gamma'):
            print(yvarname)
            print(min(yvar),max(yvar))
        ax.plot(pdate,yvar)

    #   sigma_logbeta is the standard deviation of the generating
    #   random walk, NOT the standard deviation of the estimated
    #   random effect
        if (yvarname == 'logbeta' and len(fit_files) <=4):
        #   sigma_logbeta = fit.get_est_or_init('logsigma_logbeta')
        #   plot_error(ax,pdate,yvar,sigma_logbeta,logscale=True)
            plot_error(ax,pdate,yvar,fit.diag['SElogbeta'],logscale=True)

        sn = short_name(fit.moniker)
        if (yvarname == 'logbeta'):
            mark_ends(ax,pdate,yvar,sn,'r')
        else:
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
        gfile = cv.graphics_path+yvarname+'_summary'+suffix+'.pdf'
        fig.savefig(gfile)
        print('plot saved as',gfile)
        plt.show(False)
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

    fit = Fit(cv.fit_path+'NassauNY'+ext)
    fit.plot(save=True,logscale=False)
    fit = Fit(cv.fit_path+'Miami-DadeFL'+ext)
    fit.plot(save=True,logscale=False)


def make_fit_table(ext = '.RData'):
    fit_files = glob.glob(cv.fit_path+'*'+ext)
    print('found',len(fit_files),ext,'files in',cv.fit_path)
    # mtime = os.path.getmtime(cspath) cv.NYT_counties
    #   dtime = datetime.fromtimestamp(mtime)
    #   self.updated = str(dtime.date())
    # updated from https://github.com/nytimes/covid-19-data.git

#   md_cols = ['county','N0','ntime','prop_zero_deaths','fn']
    md_cols = ['county','ntime','prop_zero_deaths','fn','C']
    es_cols = ['logsigma_logCP','logsigma_logDP','logsigma_logbeta','logsigma_logmu',
               'logsigma_logC','logsigma_logD','mbeta','mmu'] #,'mgamma']
    tt_cols = md_cols + es_cols
    header = ['County','$n$','$p_0$','$f$','$C$',
              '$\sigma_\eta_C$', '$\sigma_\eta_D$', '$\sigma_\\beta$','$\sigma_\\mu$',
              '$\sigma_{\ln I}$','$\sigma_{\ln D}$','$\\tilde{\\beta}$','$\\tilde{\\mu}$']
            #,'$\\tilde\\gamma$']

    tt = pd.DataFrame(columns=tt_cols,dtype=None)

    func = pd.DataFrame(columns=['fn'])
#   mgamma = pd.DataFrame(columns=['mgamma'])
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
    #   row['prop_zero_deaths'] = round(float(fit.get_metadata_item('prop_zero_deaths')),sigfigs)
        row['prop_zero_deaths'] = float(fit.get_metadata_item('prop_zero_deaths'))
        tt = tt.append(row,ignore_index=True)
        func = np.append(func,float(fit.get_metadata_item('fn')))
        beta = np.exp(diag['logbeta'])
        mbeta = np.append(mbeta,median(beta))
        mu = np.exp(diag['logmu'])
        mmu = np.append(mmu,mu.quantile(q=0.5))
    #   gamma = diag['gamma'
    #   mgamma = np.append(mgamma,gamma.quantile(q=0.5))

    tt['fn'] = func
#   tt['mgamma'] = mgamma
    tt['mbeta'] = mbeta
    tt['mmu'] = mmu

    mtime = os.path.getmtime(cv.NYT_counties)
    dtime = datetime.fromtimestamp(mtime)
    ft_name = cv.fit_path+'fit_table_'+str(dtime.date())

    csv = ft_name+'.csv'
    tt.to_csv(csv,index=False)
    print('Fit table data written to file',csv)


    tt = tt.sort_values(by='mbeta',ascending=True)#,inplace=True)

    for r in range(0,tt.shape[0]):
        for c in range(3,len(tt.columns)):
           if (tt.iloc[r,c] != None):
               tt.iloc[r,c] = round(float(tt.iloc[r,c]),sigfigs)
        c = 2
        tt.iloc[r,c] = round(float(tt.iloc[r,c]),sigfigs)

    row = pd.Series(None,index=tt.columns)
    row['county'] = 'Median'
    for n in tt.columns:
        if (n != 'county'):
            mn = tt[n].quantile()
            row[n] = mn
    tt = tt.append(row,ignore_index=True)


    tex = ft_name+'.tex'
    ff = open(tex, 'w')
    caption_text = "Model results. Estimating $\\beta$ and $\mu$ trends as random effects with $\gamma = 0$.\nData updated " + str(dtime.date()) + " from https://github.com/nytimes/covid-19-data.git.\n"

    ff.write(caption_text)
#   ff.write(str(dtime.date())+'\n')
    ff.write(tabulate(tt, header, tablefmt="latex_raw",showindex=False))
#   tt.to_latex(buf=tex,index=False,index_names=False,longtable=False,
#               header=header,escape=False,#float_format='{:0.4f}'.format
#               na_rep='',column_format='lrrrrrrrrrrr')
    print('Fit table written to file',tex)

def plot_dow_boxes(nG=10):
    cv.population_dat = pd.read_csv(cv.census_data_path,header=0,comment='#')
    gg = cv.population_dat

    dow_names = ['Mo','Tu','We','Th','Fr','Sa','Su']
    if nG > 5:
        rows = 2
        cols = 1
        ht = 2.25
        Cweight = 100000
        Dweight =  10000
    else:
        rows = nG
        cols = 2
        ht = 9.0/rows

    fig, ax = plt.subplots(rows,cols,figsize=(6.5,ht*rows))

    for g in range(0,nG):
    #   print(gg['county'].iloc[g])
        tmpG = Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                         code=gg['code'].iloc[g])
        tmpG.read_nyt_data('county')
        Ccount = pd.Series([0.0]*len(dow_names),index=dow_names)
        Dcount = pd.Series([0.0]*len(dow_names),index=dow_names)
        d1_cases = np.diff(tmpG.cases)
        d1_deaths = np.diff(tmpG.deaths)
        for k in range(0,len(tmpG.date)-1):            
            j = datetime.strptime(tmpG.date[k],'%Y-%m-%d').weekday()
            Ccount[j] += d1_cases[k]
            Dcount[j] += d1_deaths[k]

        Ccount = Ccount.reindex(index=dow_names)
        Dcount = Dcount.reindex(index=dow_names)
        if nG > 5:
            Ccount = (Cweight/tmpG.population)*Ccount/Ccount.sum()
            Dcount = (Dweight/tmpG.population)*Dcount/Dcount.sum()
            ax[0].bar(dow_names,Ccount)
            ax[0].set_ylabel('Cases per '+str(int(Cweight/1000))+'K')
            ax[1].bar(dow_names,Dcount)
            ax[1].set_ylabel('Deaths per '+str(int(Dweight/1000))+'K')

        else:
            Ccount = Ccount/Ccount.sum()
            Dcount = Dcount/Dcount.sum()
       
            ax[g,0].bar(dow_names,Ccount)
            ax[g,0].set_ylabel('Cases')
            ax[g,1].bar(dow_names,Dcount)
            ax[g,1].set_ylabel('Deaths')
            tx = prop_scale(ax[g,0].get_xlim(),0.1)
            ty = prop_scale(ax[g,0].get_ylim(),1.0)
            ax[g,0].text(tx,ty,tmpG.moniker,ha='center',fontsize=8)
        
  
    gfile = cv.graphics_path+'days_of_week_'+str(nG)+'.png'
    plt.savefig(gfile,dpi=300)
    plt.show()
 
        
    

        
  
def web_update():
    os.system('git -C /home/other/nytimes-covid-19-data pull -v')
    
    BC_cases_file = 'BCCDC_COVID19_Dashboard_Case_Details.csv'
    cmd = 'wget http://www.bccdc.ca/Health-Info-Site/Documents/' + BC_cases_file +\
         ' -O '+cv.cv_home+BC_cases_file
    print(cmd)
    os.system(cmd)

def make_dat_files():
    nyt_counties = pd.read_csv(cv.census_data_path,header=0,comment='#')
#   gg_filter = nyt_counties['flag'] == 1
    gg_filter = nyt_counties['flag'].str.contains('m')
    gg = nyt_counties[gg_filter]
    print(gg)

    for g in range(0,len(gg)):
        print(gg['county'].iloc[g])
        tmpG = Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                         code=gg['code'].iloc[g])
        tmpG.read_nyt_data('county')
        tmpG.write_dat_file()
        tmpG.plot_prevalence(save=True,cumulative=False, show_order_date=False)

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
    nyt_counties = pd.read_csv(cv.census_data_path,header=0,comment='#')
    gg_filter = nyt_counties['flag'].str.contains('s')
#   print(gg_filter)
    gg = nyt_counties[gg_filter]
    print(gg)
    save_path = cv.graphics_path
    cv.graphics_path = cv.cv_home+'PlotsToShare/'
    plt.rc('figure', max_open_warning = 0)
    for g in range(0,len(gg)):
        print(gg['county'].iloc[g])
        tmpG = Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                         code=gg['code'].iloc[g])
        tmpG.read_nyt_data('county')
        tmpG.plot_prevalence(save=True,signature=True,cumulative=False,
                             show_order_date=False)

    tmpG = Geography(name='Vancouver Island',enclosed_by='British Columbia',code='BC')
    tmpG.read_BCHA_data()
    tmpG.plot_prevalence(save=True,signature=True,cumulative=False,
                         show_order_date=False)

    cv.graphics_path = save_path

    os.system('git commit ~/Projects/SIR-Models/PlotsToShare/\*.png -m "Update PlotsToShare"')
    os.system('git push')


def plot_multi_prev(Gfile='top30.csv',mult = 1000,save=False):
#   gg = pd.read_csv(cv.cv_home+'top30.csv',header=0,comment='#')
    gg = pd.read_csv(cv.cv_home+Gfile,header=0,comment='#')
    #                encoding = "ISO-8859-3")
    print(gg.columns)
#   print(gg)

    key_cols = ('key','name','code')
    key = pd.DataFrame(columns=key_cols)

    fig, ax = plt.subplots(2,figsize=(6.5,9.0))
    firstDate = mdates.date2num(cv.FirstNYTDate)
    orderDate = mdates.date2num(cv.CAOrderDate)
    lastDate  = mdates.date2num(cv.EndOfTime)
    ax[0].set_ylim(0.0,100.0)
    ax[1].set_ylim(0.0, 10.0)
    for a in range(0,len(ax)):
        ax[a].set_xlim([firstDate,lastDate])
        ax[a].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax[a].xaxis.set_major_locator(mdates.MonthLocator())
        ax[a].xaxis.set_minor_locator(mdates.DayLocator())
#   Newsome's shelter in place order
        ax[a].plot((orderDate,orderDate),
            (0, ax[a].get_ylim()[1]),color='0.5',
            linewidth=3,alpha=0.5)


    ax[0].set_ylabel('New Cases'+' per '+str(mult))
    ax[1].set_ylabel('New Deaths'+' per '+str(mult))

    nG = len(gg)
    for g in range(0,nG):
        print(gg['name'][g])
        tmpG = Geography(name=gg['name'][g], enclosed_by=gg['enclosed_by'][g],
                         code=gg['code'][g])
        tmpG.read_nyt_data('county')

        delta_cases = mult*(np.diff(tmpG.cases))/tmpG.population
        delta_deaths = mult*(np.diff(tmpG.deaths))/tmpG.population
        Date = tmpG.get_pdate()
        ax[0].plot(Date[1:], delta_cases, linewidth=1)#,alpha=0.75)
        ax[1].plot(Date[1:], delta_deaths, linewidth=1)#,alpha=0.75)

        sn = short_name(tmpG.moniker)
    #   print(tmpG.name,tmpG.moniker,sn)
        kr = pd.Series((sn,tmpG.name,tmpG.code),index=key_cols)
        key = key.append(kr,ignore_index=True)

    #   if (plot_dt):
    #       mark_ends(ax,Date,delta_cases,sn,'r')
    #   else:
    #   if (~plot_dt):
    #       mark_ends(ax,Date,cases,sn,'r')


#   print(key)
    if save:
        gfile = cv.graphics_path+'counties_per_capita'+str(nG)+'.png'
        plt.savefig(gfile) #,dpi=300)
        plt.show(False)
        print('plot saved as',gfile)

    #   kfile = cv.graphics_path+'short_name_key.csv'
    #   key.sort_values(by='key',ascending=True,inplace=True)
    #   print(key)
    #   key.to_csv(kfile,index=False)
    #   print('key names saved as',kfile)

        plt.pause(5)
        plt.close()
            
    else:
        plt.show()


def update_everything():
    web_update()
    print('Finished web_update ...')
    os.system('rm -v '+ cv.dat_path + '*.dat')
    make_dat_files()
    print('Finished make_dat_files()')
    update_shared_plots()
    print('Finished update_shared_plots()')
    os.system('rm -v' + cv.fit_path + '*.RData')
    update_fits()
    print('Finished update_fits()')
    make_fit_plots()
    print('Finished fit_plots')
    make_fit_table()
    make_rate_plots('logbeta',add_doubling_time = True,save=True)
    make_rate_plots('logbeta',add_doubling_time = True,save=True,
                     fit_files=['Miami-DadeFL','HonoluluHI','NassauNY','CookIL'])
    make_rate_plots('logmu',save=True)
    print('Finished rate_plots')
    plot_DC(1000)
    print('Finished CFR plots')
    print('Finished Everything!')


# --------------------------------------------------       
print('------- here ------')
def junk_func():
    import math
    import scipy.stats as stats
    cfr = np.array(pd.read_csv('cfr500.txt'))
    logcfr = np.log(cfr+eps)
    #print(stats.describe(cfr))
    d_logcfr = stats.describe(logcfr)
    print('lmean=',d_logcfr.mean,'lvariance=',d_logcfr.variance)
    print('lmean=',type(d_logcfr.mean),'lvariance=',type(d_logcfr.variance))
    #print(stats.describe(logcfr))
    
    
    
    fig, ax = plt.subplots(2,figsize=(6.5,6.5))
    
    lmean = d_logcfr.mean[0]
    print(lmean,type(lmean),np.exp(lmean))
    lsigma = math.sqrt(d_logcfr.variance[0])
    print(lsigma,type(lsigma),np.exp(lsigma))
#   x = np.arange(lmean - 3.0*lsigma, lmean + 3.0*lsigma, 0.1)
    x = np.linspace(lmean - 3.0*lsigma, lmean + 3.0*lsigma, 50) 
    lpdf =stats.norm.pdf(x, lmean, lsigma)
    ax[0].hist(logcfr,50,density=True)
    ax[0].plot(x,lpdf) 
    ax[0].plot((lmean,lmean),ax[0].get_ylim())
    
    
    print('----------------')
    weights = np.ones_like(cfr) / len(cfr)
    nx, xbins, ptchs  = ax[1].hist(cfr,50,weights=weights)
    #print('nx',nx)
    #print('xbins',xbins)
    #print('ptchs',ptchs)
    ex = np.exp(x)
    #weights = np.ones_like(np.exp(lpdf))/len(lpdf)
    #ax[1].plot(ex, np.exp(lpdf)*weights)
    
    mean = np.log(lmean*lmean/np.sqrt(lmean*lmean-lsigma*lsigma))
    print(mean,type(mean),np.exp(mean))
    sigma = np.log(1.0-lsigma*lsigma/lmean*lmean)
    print(sigma,type(sigma),np.exp(sigma))
    mode = np.exp(mean-sigma*sigma)
    print('mode',mode,np.log(mode))
    pdf = lpdf/lpdf.sum()
    ax[1].plot(ex,pdf)
    ax[1].plot((np.exp(lmean),np.exp(lmean)),ax[1].get_ylim())
#   ax[1].plot((mode,mode),ax[1].get_ylim())
    
    
    plt.show()

#import math
#import scipy.stats as stats
#cfr = np.array(pd.read_csv('cfr500.txt')
##print(cfr)
#print(len(cfr))
#d_cfr = stats.describe(cfr)
#print('mean=',d_cfr.mean,'variance=',d_cfr.variance)
#print('mean=',type(d_cfr.mean),'variance=',type(d_cfr.variance))

#junk_func()
#mean = d_cfr.mean[0]
#sigma = math.sqrt(d_cfr.variance[0])
#print(mean,sigma)
#x = np.linspace(mean - 3.0*sigma, mean + 3.0*sigma, 50) 
#pdf =stats.norm.pdf(x, mean, sigma)
#pdfsum = sum(pdf)
#fig, ax = plt.subplots(1,figsize=(6.5,6.5))
#ax.plot(x,pdf/pdf.sum())
#ax.plot((mean,mean),ax.get_ylim())
#plt.show()

# -------------------------------------------------

#unique()

#tgeog = Geography(name='Santa Clara',enclosed_by='California',code='CA')
#tgeog.read_nyt_data('county')
#tgeog.print_metadata()
#tgeog.print_data()
#tgeog.plot_prevalence(save=True,cumulative=False, show_order_date=False,signature=True)

#cv.fit_path = cv.fit_path+'constrainID/'
#tfit = Fit(cv.fit_path+'CookIL.RData') #'Los Angeles','California','CA','ADMB')
#tfit.print_metadata()
#tfit.plot(save=True,logscale=True)

#tfit = Fit(cv.fit_path+'AlamedaCA.RData')
#tfit.plot_CMR()
#tfit.plot(save=False,logscale=False)

#update_everything()
#web_update()
#make_dat_files()
#update_fits()
#update_shared_plots()
#plot_DC(10) #00)

#make_nyt_census_dat()

#cv.fit_path = cv.fit_path+'unconstrained/'
#update_fits()
#make_fit_table()
#make_fit_plots()
#make_rate_plots('logbeta',add_doubling_time = True,save=True)
#make_rate_plots('logbeta',add_doubling_time = True,save=True,fit_files=['Miami-DadeFL','HonoluluHI','NassauNY','CookIL'])
#make_rate_plots('logmu',save=True)
#make_rate_plots('gamma',save=True)
#plot_multi_prev(save=True,mult=100000)
#plot_multi_prev(Gfile='top500.csv',save=True,mult=100000)

#test = Geography(name='Nassau',enclosed_by='New York',code='NY')
#test = Geography(name='Miami-Dade',enclosed_by='Florida',code='FL')
#test = Geography(name='New York City',enclosed_by='New York',code='NY')
#test.read_nyt_data()
#test.write_dat_file()
#test.print_metadata()
#test.plot_per_capita_curvature()
#test.plot_prevalence(save=False,cumulative=False, show_order_date=False)

plot_dow_boxes()
#plot_multi_per_capita(plot_dt=False,save=True)
#get_mu_atD1()


#verify anomaly in Nassau County death tally
#print(cv.NYT_home,cv.dat_path)
#cv.NYT_home = '/home/jsibert/Downloads/'
#cv.NYT_counties = cv.NYT_home + 'us-counties.csv'
#cv.dat_path = cv.NYT_home
#print(cv.NYT_home,cv.dat_path)

#test = Geography(name='Plumas',enclosed_by='California',code='CA')
#test.read_nyt_data()
#test.plot_prevalence(save=False,signature=True,cumulative=False,show_order_date=False)
#test.write_dat_file()

#web_update()
#BCtest = Geography(name='Vancouver Island',enclosed_by='British Columbia',code='BC')
#BCtest.read_BCHA_data()
#BCtest.print_metadata()
#BCtest.print_data()
#BCtest.plot_prevalence(save=True,signature=True,cumulative=False, show_order_date=False)


#junk_func()
#make_SD_tab()

def make_cfr_histo_ts(nG = 1000,save=True):
    firstDate = date(2020,1,1)
#   print(firstDate, mdates.date2num(firstDate))
    lastDate = datetime.fromtimestamp(os.path.getmtime(cv.NYT_home+'us-counties.csv')).date()
#   print('lastDate:',lastDate)
    nextDate = firstDate
    p = 0
    tdate = int(mdates.date2num(firstDate))
    hdate = [tdate]
    period_labels = [mdates.num2date(tdate).strftime('%b')]
    Period_Labels = [mdates.num2date(tdate).strftime('%B')]
    
    while nextDate < lastDate:
        p += 1
        nextDate = nextDate + relativedelta(months=+1)
    #   print(p,nextDate, mdates.date2num(nextDate))
        if (p>1):
            tdate = int(mdates.date2num(nextDate))
            hdate.append(tdate)
            period_labels.append(mdates.num2date(tdate).strftime('%b'))
            Period_Labels.append(mdates.num2date(tdate).strftime('%B'))
    
    
    
    #print(hdate)
    #print(period_labels)
    periods = pd.DataFrame(columns=('pd','pl','PL'))
    periods['pd']=hdate
    periods['pl']=period_labels
    periods['PL']=Period_Labels
    print(periods)
 
    cfrG = pd.DataFrame(columns=hdate[0:len(hdate)-1])
#   print('cfrG 0:')
#   print(cfrG)
    gg = pd.read_csv(cv.census_data_path,header=0,comment='#')
    print('Processing',nG,'geographies')
    nobs = 0
    for g in range(0,nG):
        print(g,gg['county'].iloc[g])
        tmpG = Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                         code=gg['code'].iloc[g])
        tmpG.read_nyt_data('county')
        tmpG.get_pdate()

        cf = pd.DataFrame(index=tmpG.get_pdate(),columns=('cases','deaths','pdate'))
        cf['cases'] = tmpG.cases
        cf['deaths'] = tmpG.deaths
        cf['pdate'] = np.int64(tmpG.pdate)
    #   print('cf:')
    #   print(cf)

        row = pd.Series(0.0,index=cfrG.columns)
        for t in range(0,len(cfrG.columns)-1):
            csum = float(cf['cases'] [hdate[t]:hdate[t+1]].sum())
            dsum = float(cf['deaths'][hdate[t]:hdate[t+1]].sum())
        #   print(hdate[t],hdate[t+1],csum,dsum)
            if (csum > 0.0):
                row[hdate[t]] = dsum/csum
                nobs += 1
            else:
                row[hdate[t]] = 0.0

        cfrG = cfrG.append(row, ignore_index=True)

#   print('cfrG nG:')
#   print(cfrG)

#   print(np.ones_like(cfrG))
#   print(cfrG.shape,cfrG.size)
#   print('cfrG.min:')
#   print(cfrG.min().max())
#   print('cfrG.max:')
#   print(cfrG.max())

    logcfrG = pd.DataFrame(columns=hdate[0:len(hdate)-1])
    logcfrG =np.log(cfrG + eps)
#   print('logcfrG"')
#   print(logcfrG)
#   df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)] 
#   logcfrG[logcfrG.replace([np.inf, -np.inf], np.nan,inplace=True).notnull().all(axis=1)] 
#   df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
#   logfcfrG = pd.DataFrame(logcfrG.replace([np.inf, -np.inf], np.nan,inplace=True)).dropna(axis=1)
#   print(logcfrG)

    bins = np.linspace(0.0,0.1,50)
#   print(bins)
    bins[0] =  0.001 #0.1*cfrG.min().max()
#   print(bins)
    fig, ax = plt.subplots(2,1,figsize=(6.5,6.5))
#   ax[0].set_ylim(0.0,0.2)
#   ax[1].set_ylim(0.0,0.2)
#   gweights = np.ones_like(cfrG) / float(cfrG.size)
    for k,p in enumerate(cfrG.columns):
        weights = np.ones_like(cfrG[p]) / float(cfrG[p].size)
        phist, bin_edges = np.histogram(cfrG[p], bins=bins, weights=weights,
                                        density=False)
        psum = phist.sum()
    #   print('psum:',psum)
        ax[0].plot(bin_edges[:-1],phist,linewidth=2)
#       mark_ends(ax[0],bin_edges[:-1],phist,period_labels[k])
        mark_peak(ax[0],bin_edges[:-1],phist,str(period_labels[k]))
 
    axlim = ax[0].get_xlim()
    tx = axlim[0] + 0.95*(axlim[1]-axlim[0])
    aylim = ax[0].get_ylim()
    ty = aylim[0]+0.9*(aylim[1]-aylim[0])
    ax[0].text(tx,ty,' n = '+str(nG),ha='right',va='center')

    logbins = np.log(bins)#+eps)
    for k,p in enumerate(cfrG.columns):
        weights = np.ones_like(logcfrG[p]) / float(logcfrG[p].size)
        phist, bin_edges = np.histogram(logcfrG[p], density=False, bins=logbins, weights=weights)
#       psum = phist.sum()
#       print('psum:',psum)
        ax[1].plot(bin_edges[:-1],phist,linewidth=2)
#       mark_ends(ax[1],bin_edges[:-1],phist,period_labels[k])
        mark_peak(ax[1],bin_edges[:-1],phist,period_labels[k])
 
    axlim = ax[1].get_xlim()
    tx = axlim[0] + 0.95*(axlim[1]-axlim[0])
    aylim = ax[1].get_ylim()
    ty = aylim[0]+0.9*(aylim[1]-aylim[0])
    ax[1].text(tx,ty,' n = '+str(nG),ha='right',va='center')

    if (save):
        gfile = cv.graphics_path+'monthly_CFR_histo_'+str(nG)+'.png'
        plt.savefig(gfile,dpi=300)
        print('Plot saved as',gfile)

    plt.show()#block=False)

    if (1):
        sys.exit(1)
    figb, bx = plt.subplots(4,3,figsize=(9.0,6.5))
    bxf = bx.flatten('C')
#   print(bxf)
    for k,p in enumerate(cfrG.columns):
        weights = np.ones_like(cfrG[p]) / float(cfrG[p].size)
        phist, bin_edges = np.histogram(cfrG[p], bins=bins, weights=weights,
                                        density=False)
        bxf[k].bar(bin_edges[:-1],phist)
        bxf[k].plot(bin_edges[:-1],phist,linewidth=2)
        axlim = bxf[k].get_xlim()
        tx = axlim[0] + 0.9*(axlim[1]-axlim[0])
        aylim = bxf[k].get_ylim()
        ty = aylim[0]+0.9*(aylim[1]-aylim[0])
        bxf[k].text(tx,ty,period_labels[k],ha='left',va='center')


    plt.show()#block=False)

#make_cfr_histo_ts()
