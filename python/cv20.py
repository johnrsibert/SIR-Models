#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taking a more OO approach to the code in cv19.py

Created on Thu Jul  2 09:04:11 2020

@author: jsibert
"""

"""
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
"""

#import js_covid as cv
from config import *
from Geography import *


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

# ---------------- global utility functions ---------------------------

def Strptime(x):
    """
    wrapper for datetime.strptime callable by map(..)
    """
    y = datetime.strptime(x,'%Y-%m-%d')
    return(y)

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

def isNaN(num):
    return num != num

def median(x):
    mx = x.quantile(q=0.5)
    return float(mx)
# -----------  class definitions--------------       
# moved to Geography.py
# moved to fit.py
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

def plot_DC(glist=[5,100], save=True):

    def vline(ax, x, label=None, ylim=None, pos='center'):
        if ylim is None:
           ylim = ax.get_ylim()
        ax.plot((x,x), ylim, linewidth=1, linestyle=':')
        c = ax.get_lines()[-1].get_color()
        ax.text(x,ylim[1], label, ha=pos,va='bottom', linespacing=1.8,
                fontsize=8, color=c)

    def mark_points(coll,ax,x,y,label,end='b',spacer=' '):
        c = coll.get_facecolor()[0]
        if ( (end =='l') | (end == 'b')):
            mark = ax.text(x[0],y[0],label+spacer,ha='right',va='center',fontsize=8,
                    color=c)

        if ( (end =='r') | (end == 'b')):
            i = len(x)-1
            mark = ax.text(x[i],y[i],spacer+label,ha='left',va='center',fontsize=8,
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
            mark_ends(a,xr,yr,rstr,'r',' ')

    def save_plot(plt,save,n,what):
        if save:
            gfile = cv.graphics_path+'CFR_'+what+'_'+str(n)+'.png'
            plt.savefig(gfile,dpi=300)
            plt.show(False)
        #   plt.pause(5)
            
            print('Plot saved as',gfile)
        else:
            plt.show()
        plt.close()

    def set_axes(ax):
        ax.set_yscale('log')
        ax.set_ylabel('Deaths')
        ax.set_ylim(1,1e5)
        ax.set_xscale('log')
        ax.set_xlabel('Cases')
        ax.set_xlim(10,1e6)



    plt.rcParams["scatter.marker"] = '.'
    plt.rcParams["lines.markersize"] = 3


    gg = pd.read_csv(cv.census_data_path,header=0,comment='#')
    for i,nG in enumerate(glist):

        fig, ax = plt.subplots(1,figsize=(6.5,4.5))
        set_axes(ax)
        recent = pd.DataFrame(columns = ('moniker','cases','deaths','cfr'))
        print('Processing',nG,'geographies')
        for g in range(0,nG):
            print(g,gg['county'].iloc[g])
            tmpG = Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                             code=gg['code'].iloc[g])
            tmpG.read_nyt_data('county')
        #   plot scatter of all in tmpG geography    
            coll = ax.scatter(tmpG.cases,tmpG.deaths)
            if (nG < 6):
                sn = short_name(tmpG.moniker)
                mark_points(coll,ax,tmpG.cases,tmpG.deaths,sn,'r')

            nt = len(tmpG.cases)-1 # index of most recent report
            cfrt = tmpG.deaths[nt]/tmpG.cases[nt]
            row = pd.Series(index=recent.columns)
            row['moniker'] = tmpG.moniker
            row['cases'] = tmpG.cases[nt]
            row['deaths'] = int(tmpG.deaths[nt])
            row['cfr'] = cfrt
            recent = recent.append(row, ignore_index=True)


        logxlim = np.log(ax.get_xlim())
        tx = np.exp(logxlim[0]+0.05*(logxlim[1]-logxlim[0]))
        logylim = np.log(ax.get_ylim())
        ty = np.exp(logylim[0]+0.95*(logylim[1]-logylim[0]))
        note = '{0} Counties; {1:,} Cases; {2:,} Deaths'.format(nG,recent['cases'].sum(),recent['deaths'].sum())
        ax.text(tx,ty,note ,ha='left',va='center',fontsize=10)
        plot_cmr(ax, [0.5,1.0,2.0,4.0,8.0])
        add_data_source(fig)
        save_plot(plt,save,nG,'all')

    recent = recent.sort_values(by='cases',ascending=False)
    recent.to_csv('recent_cfr.csv',index=False)

    fig, ax = plt.subplots(1,figsize=(6.5,4.5))
    nbins = 50
    xticks = np.linspace(0.0,0.08,num=9)
    ax.set_xlim(xticks[0],xticks[len(xticks)-1])
    ax.set_xlabel('Most Recent Case Fatality Ratio')
    ax.set_ylabel('Number')
    ax.set_xticks(xticks)
#   hist,edges,patches = ax.hist(recent['cfr'],nbins)
    weights = np.ones_like(recent['cfr']) / len(recent['cfr'])
    hist,edges,patches = ax.hist(recent['cfr'],nbins,weights=weights,density=True)

    param = stats.lognorm.fit(recent['cfr'])
    pedges  = np.linspace(0.0,0.1,500)
    pdf = stats.lognorm.pdf(pedges,param[0],param[1],param[2])
#   prob = len(recent['cfr'])*pdf/sum(pdf)
    ax.plot(pedges,pdf,linewidth=1,linestyle='--')

    median = np.median(recent['cfr'])
    Q95 = np.quantile(recent['cfr'],q=0.975)
    mode = pedges[pd.Series(pdf).idxmax()]
    ylim = ax.get_ylim()
    ylim = (ylim[0],0.95*ylim[1])
    vline(ax,median,'Median',ylim=ylim,pos='left')
    vline(ax,mode,'Mode',ylim=ylim,pos='right')
    vline(ax,Q95,'97.5%',ylim=ylim)

    xlim = ax.get_xlim()
#   tx = xlim[0]+0.95*(xlim[1]-xlim[0])
    tx = xlim[1]
    ylim = ax.get_ylim()
    ty = ylim[0]+0.90*(ylim[1]-ylim[0])
    note = '{0} Counties; {1:,} Cases; {2:,} Deaths'.format(nG,recent['cases'].sum(),recent['deaths'].sum())
    ax.text(tx,ty,note,ha='right',va='center',fontsize=10)
    add_data_source(fig)

    save_plot(plt,save,nG,'hist')



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
            mark_ends(ax,pdate,yvar,sn,'b')
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
        gfile = cv.graphics_path+yvarname+'_summary'+suffix+'.png'
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
    fit = Fit(cv.fit_path+'New_York_CityNY'+ext)
    fit.plot(save=True,logscale=False)
    fit = Fit(cv.fit_path+'Los_AngelesCA'+ext)
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
              '$\sigma_{\eta_C}$', '$\sigma_{\eta_D}$', '$\sigma_\\beta$','$\sigma_\\mu$',
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

def plot_dow_boxes(nG=5):
    cv.population_dat = pd.read_csv(cv.census_data_path,header=0,comment='#')
    gg = cv.population_dat

    dow_names = ['Mo','Tu','We','Th','Fr','Sa','Su']
    week = [0,1,2,3,4,5,6]
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
    ecol = 'black' # '#008fd5' # 'mediumblue'
    fcol = '#008fd5' # 'cornflowerblue'

    allCcount =pd.DataFrame(columns=week)
    allDcount =pd.DataFrame(columns=week)
    population = pd.Series([]*nG)

    for g in range(0,nG):
    #   print(gg['county'].iloc[g])
        tmpG = Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                         code=gg['code'].iloc[g])
        tmpG.read_nyt_data('county')
        Ccount = pd.Series([0.0]*len(dow_names)) #,index=dow_names)
        Dcount = pd.Series([0.0]*len(dow_names)) #,index=dow_names)
        d1_cases = np.diff(tmpG.cases)
        d1_deaths = np.diff(tmpG.deaths)
        for k in range(0,len(tmpG.date)-1):            
            j = datetime.strptime(tmpG.date[k],'%Y-%m-%d').weekday()
            Ccount[j] += d1_cases[k]
            Dcount[j] += d1_deaths[k]

        population[g] = tmpG.population
    #   print(g,tmpG.moniker,tmpG.population,Ccount.sum(),Dcount.sum())
    #   counties with zero reported deaths cause NaNs
        Ccount = Ccount/(Ccount.sum()+eps)
        Dcount = Dcount/(Dcount.sum()+eps)
        allCcount = allCcount.append(Ccount,ignore_index=True)
        allDcount = allDcount.append(Dcount,ignore_index=True)
    #   verify final day of week is correct with respect to calander
    #   k  = len(tmpG.date)-1
    #   print(tmpG.date[k],j,dow_names[j])

        if nG <= 5:
            ax[g,0].bar(week,Ccount,tick_label=dow_names,color=fcol,edgecolor=ecol)
            ax[g,0].set_ylabel('Cases')
            ax[g,1].bar(week,Dcount,tick_label=dow_names,color=fcol,edgecolor=ecol)
            ax[g,1].set_ylabel('Deaths')
            tx = prop_scale(ax[g,0].get_xlim(),0.1)
            ty = prop_scale(ax[g,0].get_ylim(),1.0)
            ax[g,0].text(tx,ty,tmpG.moniker,ha='center',fontsize=8)

        
    if nG > 5:
    #   for g in range(0,nG):
    #       allCcount.iloc[g] = (Cweight/population[g])*allCcount.iloc[g]
    #       allDcount.iloc[g] = (Dweight/population[g])*allDcount.iloc[g]
        title = str(nG)+' most populous US counties'
        fig.text(0.5,0.9,title,ha='center',va='bottom')
        ax[0].boxplot(allCcount.transpose(),labels=dow_names)
        ax[0].set_ylabel('Cases') # per '+str(int(Cweight/1000))+'K')
        ax[1].boxplot(allDcount.transpose(),labels=dow_names)
        ax[1].set_ylabel('Deaths') # per '+str(int(Dweight/1000))+'K')
        ax[1].set_ylim(0.0,1.0) #ax[0].get_ylim())
  
    gfile = cv.graphics_path+'days_of_week_'+str(nG)+'.png'
    plt.savefig(gfile,dpi=300)
    plt.show()
  
def web_update():
    os.system('git -C /home/other/nytimes-covid-19-data pull -v')
    
    BC_cases_file = 'BCCDC_COVID19_Dashboard_Case_Details.csv'
#               http://www.bccdc.ca/Health-Info-Site/Documents/
#                    BCCDC_COVID19_Dashboard_Case_Details.csv
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

def update_assets():
    asset_files = ['CFR_1000.png', 'logbeta_summary_2.png', 'logbeta_summary_g.png',
                   'days_of_week_5.png','days_of_week_1000.png', 
                   'CFR_all_1000.png', 'CFR_all_5.png', 'CFR_hist_1000.png',
                   'logmu_summary_g.png', 'Los_AngelesCA_prevalence.png', 
                   'New_York_CityNY_prevalence.png']
    for file in asset_files:
        cmd = 'cp -p '+cv.graphics_path+file+' '+cv.assets_path
        os.system(cmd)

    os.system('git commit ~/Projects/SIR-Models/assets/\*.png -m "Update assets"')
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
                    fit_files=['Los_AngelesCA','New_York_CityNY'])
#                   fit_files=['Miami-DadeFL','HonoluluHI','NassauNY','CookIL'])
    make_rate_plots('logmu',save=True)
    print('Finished rate_plots')
    plot_DC(glist=[5,1000], save=True)
    print('Finished CFR plots')
    update_assets()
    print('Finishing update asset directory')
    print('Finished Everything!')


# --------------------------------------------------       
print('------- here ------')
def log_norm_cfr():
    def vline(ax, y, ylim, mark):
        ax.plot((y,y), ylim)
        mark_ends(ax, (y,y), ylim, mark, 'r')

    import math
    import scipy.stats as stats
    cfr = np.array(pd.read_csv('recent_cfr.csv')['cfr'])

    bins = np.linspace(0.0,0.1,50)
    nbin = len(bins)

    fig, ax = plt.subplots(2,figsize=(6.5,6.5))

    print('1 ----------------')
    weights = np.ones_like(cfr) / len(cfr)
    hist,edges,patches = ax[0].hist(cfr,nbin,weights=weights,density=True)

    print('hist sum:',sum(hist))

    width = max(edges)/nbin
    ax[0].set_xlabel('CFR')
    ax[0].set_ylabel('Number')

    params = stats.lognorm.fit(cfr)
    tedges  = np.linspace(0.0,0.1,200)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    print('params:',params)
    print(loc,scale)
    prob = stats.lognorm.pdf(tedges,params[0],params[1],params[2])
    print('prob sum:',sum(prob))
    halfwidth = 0.5*(tedges[1]-tedges[0])
    ax[0].plot(tedges-halfwidth,prob,linewidth=2,linestyle='--')

    wparams = stats.weibull_min.fit(cfr)
    print('wparams:',wparams)
    wshape = wparams[0]
    wscale = params[2]
    print('wshape:',wshape)
    print('wscale:',wscale)
    wprob = stats.weibull_min.pdf(tedges,wparams[0],wparams[1],wparams[2])
    print('wprob sum:',sum(wprob))
    ax[0].plot(tedges-halfwidth,wprob,linewidth=3,linestyle=':')

    print('2 ----------------')
    logcfr = np.log(cfr)
    lhist,ledges = np.histogram(logcfr,bins=nbin,density=False)
    lwidth = 1.05*max(ledges)/nbin
    ax[1].bar(ledges[:-1],lhist,width=lwidth,color='0.85') #next(prop_iter)['color']
    ax[1].set_xlabel('log CFR')
    ax[1].set_ylabel('Number')

    lparam = stats.norm.fit(logcfr)
    mean = np.exp(np.mean(logcfr))
    std = np.exp(np.std(logcfr))
    median = np.exp(np.median(logcfr))
    mode = tedges[pd.Series(prob).idxmax()]-halfwidth
    Q95 = np.quantile(cfr,q=0.95)
    Q99 = np.quantile(cfr,q=0.99)
    vlim = ax[0].get_ylim()
    vline(ax[0],mean,vlim,'mean')
    vline(ax[0],mode,vlim,'mode')
    vlim = (vlim[0],0.8*vlim[1])
    vline(ax[0],Q95,vlim,'95%')
    vline(ax[0],Q99,vlim,'99%')

    lpdf = stats.norm.pdf(ledges,lparam[0],lparam[1])#,lparam[2])
#   ensure the area under the pdf agrees with the area under the bars
    lprob = len(logcfr)*lpdf/sum(lpdf)
    ax[1].plot(ledges,lprob)#,color=next(prop_iter)['color']) #color='0.7', #,color= 

    plt.savefig('log_norm_cfr.png',format='png',dpi=300)
    plt.show()

#


# -------------------------------------------------


#cv.fit_path = cv.fit_path+'constrainID/'
#tfit = Fit(cv.fit_path+'CookIL.RData') #'Los Angeles','California','CA','ADMB')
#tfit.print_metadata()
#tfit.plot(save=True,logscale=True)
#update_everything()
#web_update()
#make_dat_files()
#update_fits()
#update_shared_plots()
#update_assets()
#plot_DC([5,1000],save=True)
#make_rate_plots('logbeta',add_doubling_time = True,save=True)
#make_rate_plots('logbeta',add_doubling_time = True,save=True,
#               fit_files=['Los_AngelesCA','New_York_CityNY'])
#make_rate_plots('logmu',save=True)

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

#plot_dow_boxes(1000)
#plot_multi_per_capita(plot_dt=False,save=True)
#get_mu_atD1()


#verify anomaly in Nassau County death tally
#print(cv.NYT_home,cv.dat_path)
#cv.NYT_home = '/home/jsibert/Downloads/'
#cv.NYT_counties = cv.NYT_home + 'us-counties.csv'
#cv.dat_path = cv.NYT_home
#print(cv.NYT_home,cv.dat_path)

#BCtest = Geography(name='Vancouver Island',enclosed_by='British Columbia',code='BC')
#BCtest.read_BCHA_data()
#BCtest.print_metadata()
#BCtest.print_data()
#BCtest.plot_prevalence(save=True,signature=True,cumulative=False, show_order_date=False)


#junk_func()
#make_SD_tab()

def make_cfr_histo_ts(nG = 100,save=True):
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


tgeog = Geography(name='Los Angeles',enclosed_by='California',code='CA')
#tgeog = Geography(name='New York City',enclosed_by='New York',code='NY')
#tgeog = Geography(name='Santa Clara',enclosed_by='California',code='CA')
#tgeog = Geography(name='Harris',enclosed_by='Texas',code='TX')
tgeog.read_nyt_data('county')
tgeog.print_metadata()
tgeog.print_data()
tgeog.plot_prevalence(save=False,cumulative=False, show_order_date=False,signature=True)
