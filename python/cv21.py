#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jsibert
"""


from covid21 import config as cv
from covid21 import Geography as GG
from covid21 import Fit as FF
from covid21 import GraphicsUtilities as GU
from covid21 import CFR

from numpy import errstate,isneginf #,array
import pandas as pd
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from matplotlib import rc
import numpy as np
import os
import sys
import pyreadr
from io import StringIO 
from io import BytesIO
import base64
import scipy.stats as stats
import time


#from sigfig import round
#from tabulate import tabulate
#from collections import OrderedDict
#import glob
#import re
#import statistics

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
def isNaN(num):
    return num != num
"""
def median(x):
    mx = x.quantile(q=0.5)
    return float(mx)
"""

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

def plot_dow_boxes(nG=5):
    cv.population_dat = pd.read_csv(cv.GeoIndexPath,header=0,comment='#')
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
    nyt_counties = pd.read_csv(cv.GeoIndexPath,header=0,comment='#')
#   gg_filter = nyt_counties['flag'] == 1
    gg_filter = nyt_counties['flag'].str.contains('m')
    gg = nyt_counties[gg_filter]
    print(gg)

    for g in range(0,len(gg)):
        print(gg['county'].iloc[g])
        tmpG = GG.Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                         code=gg['code'].iloc[g])
        tmpG.read_nyt_data('county')
        tmpG.write_dat_file()
        tmpG.plot_prevalence(save=True,cumulative=False, show_order_date=False,
             show_superspreader=False,per_capita=True)

def update_fits(njob=4):
    save_wd = os.getcwd()
    print('save:',save_wd)
    print(cv.TMB_path)
    os.chdir(cv.TMB_path)
    print('current',os.getcwd())
    cmd = 'rm -fv ' + cv.fit_path + '*.RData'
    print(cmd)
    os.system(cmd)
    os.system('rm -fv make.out & ls *.out')
    os.system('touch make.out & ls *.out')
    # globs list of monikers in dat directory
    # ensure that SIR_path is set correctly
#   cmd = 'Rscript --verbose do_glob_runs.R'
    cmd = 'make -f' + cv.TMB_path + 'Makefile ' + '-j' + str(njob) + ' -Otarget >> '\
                    + cv.TMB_path + 'make.out'
    print('Starting',cmd)
    os.system(cmd)
    print('Finished',cmd)
    os.chdir(save_wd)
    print('current',os.getcwd())

def update_shared_plots():
    quartiles = GG.plot_prevalence_comp_histo(flag='500000',window=15,save=True, 
                                              signature=True)
    print('precalence quartiles:',type(quartiles))
    print(quartiles)
    print('q[0.05] =',quartiles[0.05])
    update_html()

    nyt_counties = pd.read_csv(cv.GeoIndexPath,header=0,comment='#')
    gg_filter = nyt_counties['flag'].str.contains('s')
#   print(gg_filter)
    gg = nyt_counties[gg_filter]
    print('gg:')
    print(gg)
    save_path = cv.graphics_path
    cv.graphics_path = cv.cv_home+'PlotsToShare/'
    plt.rc('figure', max_open_warning = 0)
    for g in range(0,len(gg)):
        print(gg['county'].iloc[g])
        tmpG = GG.Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                         code=gg['code'].iloc[g])
        if (gg['code'].iloc[g] == 'BC'):
            tmpG.read_BCHA_data('province')
        else:
            tmpG.read_nyt_data('county')

#       tmpG.read_nyt_data('county')
        tmpG.plot_prevalence(save=True,signature=True,cumulative=False,
                             show_order_date=True,per_capita=True,low_prev=quartiles[0.05])

    GG.plot_prevalence_comp_TS(flag='m',low_prev=quartiles[0.05],save=True, signature=True)

#   tmpG = GG.Geography(name='Vancouver Island',enclosed_by='British Columbia',code='BC')
#   tmpG.read_BCHA_data()
#   tmpG.plot_prevalence(save=True,signature=True,cumulative=False,
#                        show_order_date=False)


    cv.graphics_path = save_path

#   os.system('git commit ~/Projects/SIR-Models/PlotsToShare/\*.png -m "Update PlotsToShare"')
#   os.system('git push')

def update_assets():
    asset_files=['prevalence_comp_TS_m.png','recent_prevalence_histo_pop.png',
                 'New_York_CityNY_prevalence.png','Los_AngelesCA_prevalence.png',
                 'CFR_all_5.png','CFR_all_0000.png','CFR_hist_all_recent.png',
                 'logbeta_summary_g.png','logbeta_summary_2.png','logmu_summary_g.png',
                 'CFR_quantiles_boxplot.png', 'days_of_week_5.png','days_of_week_1000.png']

#   asset_files = ['recent_prevalence_histo_pop.png','prevalence_comp_TS_m.png',
#                  'logbeta_summary_2.png', 'logbeta_summary_g.png',
#                  'days_of_week_5.png','days_of_week_1000.png', 
#                  'CFR_all_0000.png', 'CFR_all_5.png', 'CFR_hist_all_recent.png',
#                  'logmu_summary_g.png', 'Los_AngelesCA_prevalence.png', 
#                  'New_York_CityNY_prevalence.png','CFR_quantiles_boxplot.png']
    for file in asset_files:
        cmd = 'cp -pvf '+cv.graphics_path+file+' '+cv.assets_path
        os.system(cmd)

#   os.system('git commit ~/Projects/SIR-Models/assets/\*.png -m "Update assets"')
#   os.system('git push')

def plot_multi_prev(Gfile='top30.csv',mult = 10000,save=False):
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
    #       GU.mark_ends(ax,Date,delta_cases,sn,'r')
    #   else:
    #   if (~plot_dt):
    #       GU.mark_ends(ax,Date,cases,sn,'r')


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

def update_html():
    index_md = cv.Jon_path+'index.md'
#   print(index_md)

    tmp = cv.Jon_path+'tmp.md'
#   print(tmp)
 
#   cmd = 'git -C ' + cv.Jon_path + ' pull'
#   print(cmd)
#   os.system(cmd)
    table = cv.assets_path+'recent_prevalence_histo_pop.html'
#   print(table)
    cmd = 'cp -fv '+index_md + ' ' +index_md +'.bak'
#   print(cmd)
    os.system(cmd)

    cmd =   "sed -e '/<!---START TABLE--->/,/<!---END TABLE--->/!b' " 
#   print(cmd)
    cmd = cmd + "-e '/<!---END TABLE--->/!d;r"+table+"' "
#   print(cmd)
    cmd = cmd + "-e 'd' "+index_md+" > "+tmp
#   print(cmd)
    os.system(cmd)

    cmd = 'cp -fv '+tmp + ' ' +index_md
#   print(cmd)
    os.system(cmd)
    print('Finished update html table in index.md')

def update_everything(do_fits = True):
    web_update()
    print('Finished web_update ...')
    os.system('rm -v '+ cv.dat_path + '*.dat')
    make_dat_files()
    print('Finished make_dat_files()')
    update_shared_plots()
    print('Finished update_shared_plots()')
    CFR.plot_DC_scatter(save=True)
    CFR.plot_recent_CFR(save=True)
    print('Finished CFR plots')
#   GG.plot_prevalence_comp_TS(flag='m',save=True, signature=True)
#   GG.plot_prevalence_comp_histo(flag='500000',window=14,save=True, signature=True)
    print('Finished prevealence comp plots')

    update_assets()
    print('Finished update asset directory')

    if (do_fits):
        update_fits()
        print('Finished update_fits()')
        FF.make_fit_plots()
        print('Finished fit_plots')
        FF.make_fit_table()
        print('Finished fit table')
        FF.make_rate_plots('logbeta',show_doubling_time = True, save=True)
        FF.make_rate_plots('logbeta',show_doubling_time = True, save=True,
                    fit_files=['Los_AngelesCA','New_York_CityNY'])
    #                   fit_files=['Miami-DadeFL','HonoluluHI','NassauNY','CookIL'])
        FF.make_rate_plots('logmu',save=True)
        print('Finished rate_plots')

    print('Finished Everything!')


def log_norm_cfr():
    def vline(ax, y, ylim, mark):
        ax.plot((y,y), ylim)
        GU.mark_ends(ax, (y,y), ylim, mark, 'r')

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

def git_commit_push():
    os.system('git commit ~/Projects/SIR-Models/PlotsToShare/\*.png -m "Update PlotsToShare"')
    os.system('git commit ~/Projects/SIR-Models/assets/\*.png -m "Update assets"')
    os.system('git push')

#   cmd = 'git -C ' + cv.Jon_path + " commit index.md -m 'update table'"
#   print(cmd)
#   os.system(cmd)
#   cmd = 'git -C ' + cv.Jon_path + ' push'
#   print(cmd)
#   os.system(cmd)


def _NOT_CFR_comp(nG=5):
    d1 = cv.nyt_county_dat['date'][0]
    d2 = cv.nyt_county_dat['date'].iloc[-1]
    date_list = pd.DatetimeIndex(pd.date_range(d1,d2),freq='D')
    print('processing',nG,'geographies and',len(date_list),'dates:')
    print(d1,d2)

    dat = cv.nyt_county_dat 
    dat['fips'] = dat['fips'].fillna(0).astype(np.int64)
    NYC_mask = dat['county'] == 'New York City' 
    dat.loc[NYC_mask,'fips'] = 36999

    # CFR by geography
    gCFR = pd.Series(index=np.arange(0,nG), dtype='float64')
    gcases = pd.DataFrame(0.0,columns=np.arange(0,nG), index = date_list)
    gdeaths = pd.DataFrame(0.0,columns=np.arange(0,nG), index = date_list)
    CFR = pd.DataFrame(0.0,columns=np.arange(0,nG), index = date_list)

    gg = cv.GeoIndex

    for g in range(0,nG):
        fips = gg['fips'].iloc[g]
        print(g,':',gg['county'].iloc[g] ,fips)
    #   print(gcases[g])
        fips_mask = dat['fips'] == fips
    #   fips_entries = fips_mask.value_counts()[1]
    #   print('fips_entries:', fips_entries)
   
        gindex =  dat[fips_mask].index
        print(gindex)
        for k in gindex:
        #   print('k:',k)
            tmp = dat.loc[k]
            date = tmp['date']
            gcases.loc[date,g] = gcases.loc[date,g]+tmp['cases']
            gdeaths.loc[date,g] = gdeaths.loc[date,g]+tmp['deaths']


#   with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            # more options can be specified also
    
    print(gdeaths)
    print(gcases)

    for date in pd.DatetimeIndex(date_list):
    #   print(date,gdeaths.loc[date],gdeaths.loc[date].sum())
        CFR.loc[date]  = gdeaths.loc[date]/(gcases.loc[date]+1e-8) + 1e-8
    #   CFR.loc[date]  = gdeaths.loc[date].sum()/(gcases.loc[date].sum()+1e-8)
    #   print('    CFR    ',CFR.loc[date]) 
 
#   print(CFR)
    file_name = 'CFR'+str(nG)+'.csv'
    csv = open(file_name,'w')
#   csv.write(str(nG)+'\n')
    CFR.to_csv(csv,index=True)

    print('CFR for',nG,'geographies and',len(date_list),'written to',file_name)

def CFR_comp(nG=5):
    dat = cv.nyt_county_dat 
    dat['fips'] = dat['fips'].fillna(0).astype(np.int64)
    NYC_mask = dat['county'] == 'New York City' 
    dat.loc[NYC_mask,'fips'] = 36999

    # find first death
    k = 0
    while dat['deaths'][k]< 1.0:
        k += 1
    d1 = cv.nyt_county_dat['date'].iloc[k]
    d2 = '2020-10-01' #cv.nyt_county_dat['date'].iloc[-1]
#   d1 = cv.nyt_county_dat['date'][0]
#   d2 = cv.nyt_county_dat['date'].iloc[-1]
    date_list = pd.DatetimeIndex(pd.date_range(d1,d2),freq='D')
    print('processing',nG,'geographies and',len(date_list),'dates:')
    print(d1,d2)
    print(date_list)
    print(date_list[0])
    size = nG*len(date_list)
    print('Computing',size,'CFR estimates for',nG,'geographies and',len(date_list),'dates')
#   if (1): sys.exit(1)

    cases = pd.DataFrame(0.0,columns=np.arange(0,nG), index = date_list)
    deaths = pd.DataFrame(0.0,columns=np.arange(0,nG), index = date_list)

    gg = cv.GeoIndex

    for g in range(0,nG):
        fips = gg['fips'].iloc[g]
        print(g,':',gg['county'].iloc[g] ,fips)
    #   print(cases[g])
        fips_mask = dat['fips'] == fips
        fips_entries = fips_mask.value_counts()[1]
    #   print('fips_entries:', fips_entries)
   
        gindex =  dat[fips_mask].index
    #   print(gindex)
        for k in gindex:
        #   print('k:',k)
            tmp = dat.loc[k]
        #   print(k,tmp)
        #   print(tmp['date'],type(tmp['date']))
            date = tmp['date']
        #   date = datetime.strptime(tmp['date'],'%Y-%m-%d')
        #   print('date:',date)
            if date in date_list:
                cases.loc[date,g]  = tmp['cases']
                deaths.loc[date,g] = tmp['deaths']


#   with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            # more options can be specified also
    
    print(deaths)
    print(cases)

    dcases = pd.DataFrame(0.0,columns=np.arange(0,nG), index = date_list)
    ddeaths = pd.DataFrame(0.0,columns=np.arange(0,nG), index = date_list)

    for g in range(0,nG):
        dcases[g][1:]  = np.diff(cases[g])
        ddeaths[g][1:] = np.diff(deaths[g])

    sddeaths = pd.DataFrame(ddeaths.stack())#,columns=('date','G','ddeaths'))
    print(sddeaths)
#   print(sddeaths.shape)
#   print(sddeaths.drop('G',axis=1))
   

#   print(cases.join(deaths,lsuffix='cases',rsuffix='deaths'))
    '''
    cases  = np.zeros(size)
    dcases  = np.zeros(size)
    deaths = np.zeros(size)
    dates  = ['']*size
#   if (1): sys.exit(1)

    gg = cv.GeoIndex

    d = 0
    for dd in range(0,len(date_list)):
        date = date_list[dd].strftime('%Y-%m-%d')
        date_mask = dat['date'] == date

        print('------- d =',d)
        for g in range(0,nG):
            fips = gg['fips'].iloc[g]
            print(g,':',gg['county'].iloc[g] ,fips)
            fips_mask = dat['fips'] == fips
   
            tmp = dat[date_mask & fips_mask]
            dates[d] = date
            if (len(tmp) > 0):
                cases[d] = tmp['cases']
                deaths[d] = tmp['deaths']
         #  else:
         #      cases[d] = 0.0
         #      deaths[d] = 0.0
            d += 1



    dcd = pd.DataFrame(columns=['date','cases','dcases','deaths'])
    dcd['date'] = dates
    dcd['cases'] = cases
    dcd['deaths'] = deaths
#   dcases[1:] = pd.Series(np.diff(dcd['cases']))
    dcd['dcases'] = dcases
   
    pd.Series
    print(dcd)
    dcd.to_csv('dcd.csv',index=True)
    '''
    if (1): sys.exit(1)

    df = pd.DataFrame(columns=['date','cases','deaths'])
    df['date'] = dates
    df['cases'] = casess
    df['deaths'] = deathss
    print(df)
    if (1): sys.exit(1)

#   for date in date_list: #pd.DatetimeIndex(date_list):
    for dd, ts in enumerate(date_list):
        date = ts.strftime('%Y-%m-%d')
    #   print(dd,ts,type(ts),date)
        if (dd > 0):
            CFR.loc[date]  = ddeaths.loc[date]/(dcases.loc[date]+1e-8)
    #   print('    CFR    ',CFR.loc[date]) 
 
    print('CFR for',nG,'geographies and',len(date_list),'days')
    print(CFR)
#   print(CFR[11])
    file_name = 'CFR'+str(nG)+'.csv'
    csv = open(file_name,'w')
#   csv.write(str(nG)+'\n')
    CFR.to_csv(csv,index=True)

    print('CFR for',nG,'geographies and',len(date_list),'days written to',file_name)

def fit_lnCFR(CFRfile,Floc=None):
    CFR = pd.read_csv(CFRfile,header=0,index_col=0)
#   print(CFR.shape)
#   print(len(CFR.index))
#   print(len(CFR.columns))
#   print(CFR)
    nG = len(CFR.columns)
    ndate = len(CFR.index)

    # CFR log-normal parameters by date
    CFRln = pd.DataFrame(columns=('shape', 'loc', 'scale',
                                  'meanR','sdR', 'meanlogR','sdlogR'),
                                  index=CFR.index)
    for date in CFR.index:
        if Floc is None:
            shape, loc, scale = stats.lognorm.fit(CFR.loc[date])
        else:
            shape, loc, scale = stats.lognorm.fit(CFR.loc[date],floc=Floc)
        meanR = np.mean(CFR.loc[date])
        sdR = np.std(CFR.loc[date])
        logR = np.log(CFR.loc[date])
        meanlogR = np.exp(np.mean(logR))
        sdlogR = np.exp(np.std(logR))
        CFRln.loc[date] = [shape, loc, scale, meanR, sdR, meanlogR, sdlogR] 

    print(CFRln)

    file_name = 'CFRstats_{}_{}.csv'.format(nG,Floc)
    csv = open(file_name,'w')
    csv.write('{} {}\n'.format(nG,Floc))
    CFRln.to_csv(csv,index=True)
    print('wrote CFNln to',file_name)

def plot_CFR_lines(CFRln_file):
    csv = open(CFRln_file)
    nG,floc = csv.readline().split(' ',1)
    CFRdesc = pd.read_csv(csv,header=0,index_col=0)
    print('floc:',floc)
    print(CFRdesc)

    fig, ax = plt.subplots(1,figsize=(6.5,4.5))
    bins  = np.linspace(0.0,0.1,100)
    ax.set_xlabel('Case Fatality Ratio')
    ax.set_ylabel('Proportion')
    for d in range(0,len(CFRdesc)):
#   for d in range(len(CFRdesc)-1,41,-1):
#   for d in range(110,80,-1):
        pdf = stats.lognorm.pdf(bins,CFRdesc['shape'][d],CFRdesc['loc'][d],CFRdesc['scale'][d])
        pdf = pdf/sum(pdf)
        ax.plot(bins,pdf,linewidth=1)
        GU.mark_peak(ax,bins,pdf,str(d))

    tx = GU.prop_scale(ax.get_xlim(),0.95)
    ty = GU.prop_scale(ax.get_ylim(),0.8)
    txt = '{} counties, floc = {}\n'.format(nG,floc)
    ax.text(tx,ty,txt,ha='right',fontsize=10)
    plt.show(block=False)

    fig, ax = plt.subplots(1,figsize=(6.5,4.5))
#   ax.set_xlabel('Date')
    ax.set_ylabel('Mean CFR')
    Date = mdates.date2num(CFRdesc.index)
    GU.make_date_axis(ax)
    ax.plot(Date,CFRdesc['meanR'])
    GU.mark_ends(ax,Date,CFRdesc['meanR'],'arithmetic','r')
    ax.plot(Date,CFRdesc['meanlogR'])
    GU.mark_ends(ax,Date,CFRdesc['meanlogR'],'log trans','r')
    
    plt.show()

 
def plot_CFR_contour(CFRln_file):
    csv = open(CFRln_file)
    nG,floc = csv.readline().split(' ',1)
    CFRdesc = pd.read_csv(csv,header=0,index_col=0)
    CFRdesc.index = pd.DatetimeIndex(CFRdesc.index,freq='D') 
    d1 = '2020-02-01'# 00:00:00'
    d2 = cv.nyt_county_dat['date'].iloc[-1]
    print(d1,d2)
    date_list = pd.DatetimeIndex(pd.date_range(d1,d2),freq='D')

    bins  = np.linspace(0.0,0.10,100)
    X = date_list
    Y = bins
    Z = pd.DataFrame(columns=bins,index=date_list)
    for d in date_list:
        pdf = stats.lognorm.pdf(bins,CFRdesc.loc[d]['shape'],
                                CFRdesc.loc[d]['loc'],CFRdesc.loc[d]['scale'])
        pdf = pdf/sum(pdf)
        pdf = pd.Series(pdf,index=bins)
        Z.loc[d] = pdf

    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval = 7))

    CS = ax.contour(X, Y, Z.transpose(),levels=np.linspace(0.0,0.3,100),linewidths=1)
    ax.clabel(CS, inline=1, fontsize=8)

    plt.show()


def plot_CFR_ridge(CFRfile):
    from joypy import joyplot
    from matplotlib import cm
    CFR = pd.read_csv(CFRfile,header=0,index_col=0)
    print('finished reading',CFRfile,CFR.shape)
    nG = len(CFR.columns)
    print(CFR.shape)
    print(nG,CFR.shape[1])
    nd = CFR.shape[0]
    print(nd)
    print(CFR)

#   ntail = 0
#   if (ntail > 0):
#       shortCFR = CFR.tail(ntail)
#   else:
#       shortCFR = CFR.iloc[58:]
    shortCFR = pd.DataFrame(CFR)

    print('building labels')
    labels = [None]*len(shortCFR)
    prev_mon = 11
    for i, d in enumerate(shortCFR.index):
        dlist = d.split('-')
    #   print(i,d,dlist,prev_mon,dlist[1])
        if (i == 0):
            prev_mon = int(dlist[1])
            dd = datetime.strptime(d,'%Y-%m-%d')
            labels[i] = dd.strftime('%Y')+' '+ dd.strftime('%b')
        elif (dlist[1] == '01' and prev_mon == 12):
            prev_mon = 1
            dd = datetime.strptime(d,'%Y-%m-%d')
            labels[i] = dd.strftime('%Y')+' '+ dd.strftime('%b')
        elif (dlist[2] == '01'): # day 1 of month
            prev_mon = int(dlist[1])
            dd = datetime.strptime(d,'%Y-%m-%d')
            labels[i] = dd.strftime('%b')
        else:
            prev_mon = int(dlist[1])


    print('building "flat" file ... very slowly')
    t0 = time.time()
    flatCFR = pd.DataFrame(columns=('date','ratio'))
    for d in shortCFR.index:
        row = pd.Series(0.0,index=flatCFR.columns)
        row['date'] = d # shortCFR['date'].iloc[r]
        for c in shortCFR.loc[d]:
            row['ratio'] = c
            flatCFR = flatCFR.append(row,ignore_index = True)

    t1 = time.time()
    total = t1-t0
    print('time for method 1 is',total)
    print(flatCFR)
#   time for method 1 is 34.769570112228 [597 rows x 30 columns]
#   time for method 1 is 126.8467698097229 [597 rows x 100 columns]

    size = CFR.size
    print('building "flat" file with',size,'elements')
    t0 = time.time()
    dates = ['']*size
    ratio = [0.0]*size
    flatCFR = pd.DataFrame(columns=('date','ratio'))
    flatCFR['date'] = date
    flatCFR['ratio'] = ratio
#   print(flatCFR*size)
    i = 0
    for d in range(0,nd):
        ndx = CFR.index[d]
        row = CFR.loc[ndx]
        for g in range(0,nG):
            flatCFR['date'][i] = ndx #CFR.index[d]
            flatCFR['ratio'][i] = row[g]
            i += 1

    t1 = time.time()
    total = t1-t0
    print('time for method 2 is',total)
#   time for method 2 is 18.012922286987305 [597 rows x 30 columns]
#   time for method 2 is 69.01189374923706 [597 rows x 30 columns]
    print(flatCFR)
#   print(flatCFR.dtypes)
    flatCFR.to_csv('flatCSV.csv',index=True)
    if (1):sys.exit(1)

    print('plotting ridgeline ... even more very slowly')
    fig,axes = joyplot(flatCFR, by='date', column='ratio', labels = labels,
                       range_style='own', overlap = 3, x_range=[0.0,0.05],
                       grid="y", linewidth=0.25, legend=False, figsize=(6.5,8.0),
                       title='Case Fatality Ratio\n'+str(nG) + ' Largest US Counties',
              #        colormap=cm.autumn_r)
                       colormap=cm.Blues_r)

    print(type(axes),len(axes),type(axes[0]))
    print('xlim: ',axes[0].get_xlim())
    print('ylim: ',axes[0].get_ylim())

    gfile = 'CFRridgeline'+str(nG)+'.png' 
    print('saving',gfile)
    plt.savefig(gfile, dpi=300)
    print('ridgeline plot saved as',gfile)
    plt.show()

def recent_prevalence(min_pop=1000000,mult=10000):
    dat = pd.read_csv(cv.NYT_counties,header=0)
    dat['fips'] = dat['fips'].fillna(0).astype(np.int64) 
    dat.loc[dat['county'] == 'New York City','fips'] = 36999
    dat['deaths'] = dat['deaths'].fillna(0).astype(np.int64)
    print(dat)
    most_recent = dat['date'].iloc[-1]
    print('most recent date:',most_recent,mdates.date2num(most_recent))
 
    previous = mdates.num2date(mdates.date2num(most_recent)-1)
#   previous = mdates.date2num(most_recent)-1
    print('previous date:',previous)
#   then = datetime.datetime.strptime(when, '%Y-%m-%d').dat
#   previous =  datetime.strptime(previous,'%Y-%m-%d').date()
#   FirstNYTDate = datetime.strptime('2020-01-21','%Y-%m-%d')
#   print('previous date:',previous)

    gndx = cv.GeoIndex.set_index(cv.GeoIndex['fips'])
    print(gndx)
    gndx.to_csv('gndx.csv',index=True)
    print('gpop',gndx['population'][36999])
    cmask = dat['cases'] > 1.0
    date_mask = dat['date'] == previous #most_recent
    fmask = dat['fips']  <= 56000 # restrict to mainland counties
    zmask = cmask & date_mask & fmask
    dat = dat[zmask]
    print(dat)

    per_capita = pd.DataFrame(index=dat.index,columns=('fips','cases','deaths','population'))

    for dd in range(0,len(dat)):
    #   print(dd,dat.iloc[dd])
        fips = dat.iloc[dd]['fips']
    #   print('fips:',fips)
        pop = gndx['population'][fips]
    #   print('pop:',pop)
        per_capita.iloc[dd]['fips']=dat.iloc[dd]['fips']
        per_capita.iloc[dd]['cases']=dat.iloc[dd]['cases']/float(pop)
        per_capita.iloc[dd]['deaths']=dat.iloc[dd]['deaths']/float(pop)
        per_capita.iloc[dd]['population']=pop#get_population(dat.iloc[dd]['fips'])

    print(per_capita)
    print(stats.describe(per_capita['cases']))

    fig, ax = plt.subplots(1,figsize=(6.5,4.5))

    bins = np.linspace(0.0,0.25,50)
    print(bins)
    nbin = len(bins)
    weights = np.ones_like(per_capita['cases']) / len(per_capita)
    hist,edges,patches = ax.hist(per_capita['cases'],bins,weights=weights,density=False)

#   q = np.linspace(0.0,0.9,10)
#   print(q)
#   qq = np.quantile(per_capita['cases'],bins)
#   print(qq)

#   qx = ax.twinx()
#   qx.plot(bins,qq,color='red')

    print('quantiles:')     
    qq = [0.01,0.05,0.10]
    for q in qq:
        pq = np.quantile(per_capita['cases'],q=q)
        print(q,pq)
        GU.vline(ax,pq,str(q),pos='right')
#          vline(ax,mean,note,pos='left')
 
    plt.show()
 
    
  

   
 


# --------------------------------------------------       
print('------- here ------')
print('python:',sys.version)
print('pandas:',pd.__version__)
print('matplotib:',matplotlib.__version__)

#_NOT_CFR_comp(3)
CFR_comp(3)

#plot_CFR_ridge('CFR100.csv')

#fit_lnCFR('CFR1000.csv',Floc=0.0)
#plot_CFR_lines('CFRstats_1000_0.0.csv')
#plot_CFR_contour('CFRstats_1000_0.0.csv')

#tgeog = GG.Geography(name='Santa Clara',enclosed_by='California',code='CA')
#tgeog = GG.Geography(name='Harris',enclosed_by='Texas',code='TX')
#tgeog = GG.Geography(name='Hennepin',enclosed_by='Minnesota',code='MN')
#tgeog = GG.Geography(name='Los Angeles',enclosed_by='California',code='CA')
#tgeog = GG.Geography(name='New York City',enclosed_by='New York',code='NY')
#tgeog = GG.Geography(name='Alameda',enclosed_by='California',code='CA')
#tgeog = GG.Geography(name='Pinellas',enclosed_by='Florida',code='FL')
#tgeog.read_nyt_data('county')
#tgeog.print_metadata()
#tgeog.print_data()
#tgeog.plot_prevalence(save=True, signature=True,show_superspreader=False,per_capita=True,show_order_date = True)

#tgeog.plot_prevalence(save=False, signature=True,show_superspreader=False,per_capita=True,show_order_date = True,yscale='log')#,cumulative = True)

#tmpG = GG.Geography(name='Vancouver Island',enclosed_by='British Columbia',code='BC')
#tmpG.read_BCHA_data()
#tmpG.plot_prevalence(save=False,signature=True,cumulative=False,
#                     per_capita=True,show_order_date=False)


#web_update()
#make_dat_files()
#update_fits()
#FF.make_fit_plots()
#FF.make_rate_plots('logbeta',show_doubling_time = False, save=False, 
#                  #show_order_date = False,
#                  fit_files=['Los_AngelesCA','New_York_CityNY'])
#                  fit_files=['BrowardFL', 'NassauNY', 'MiddlesexMA',
#                             'MaricopaAZ', 'New_York_CityNY',
#                             'SuffolkNY', 'Miami-DadeFL'])
#FF.make_fit_table(path='obs_error')
#FF.make_fit_table(model_name = 'rrSIR')

#tfit = FF.Fit(cv.fit_path+'Los_AngelesCA.RData')
#tfit.plot(save=True,logscale=True,show_doubling_time=True)
#FF.make_rate_plots('logbeta',show_doubling_time = True,save=True)
#FF.make_rate_plots('logbeta',show_doubling_time = True, save=True,
#                   fit_files=['Los_AngelesCA','New_York_CityNY'])
#FF.make_rate_plots('logmu',save=True)
#update_assets()

#update_everything()#do_fits=False)
#update_html()
#git_commit_push()

#GG.plot_prevalence_comp_TS(flag='H',save=True, signature=True)
#GG.plot_prevalence_comp_TS(flag='m',save=True, signature=True)
#GG.plot_prevalence_comp_TS(flag='500000',save=True, signature=True)
#GG.plot_prevalence_comp_histo(flag='500000',window=15,save=True, signature=True)
#update_shared_plots()
#CFR.plot_recent_CFR(save=True)


#update_shared_plots()
#qq=GG.plot_prevalence_comp_histo(flag='500000',window=15,save=True, signature=True)
#GG.plot_prevalence_comp_TS(flag='m',low_prev=qq[0.05],save=True, signature=True)
#update_assets()
