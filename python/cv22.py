#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jsibert
"""
from numpy import errstate, isneginf  # ,array
from io import StringIO
import time
import scipy.stats as stats
import base64
from io import BytesIO
import pyreadr
import sys
import os
import numpy as np
from matplotlib import rc
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from datetime import date, datetime, timedelta
import pandas as pd
from covid21 import config as cv
from covid21 import vax as VV
from covid21 import CFR
from covid21 import GraphicsUtilities as GU
from covid21 import Fit as FF
from covid21 import Geography as GG
#import ipdb
# ipdb


#from sigfig import round
#from tabulate import tabulate
#from collections import OrderedDict
#import glob
#import re
#import statistics


def foo(bar):
    """


    Parameters
    ----------
    bar : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """


def make_nyt_census_dat():
    """
    Generate file with census population estimates 
    and county names  updated from https://github.com/nytimes/covid-19-data.git
    add two byte state postal codes
    add 'flag' field for selecting favorites
    add population estimate for New York City as per NYT practice
    """
    census_dat_file = cv.cv_home+'co-est2019-pop.csv'
    census_dat = pd.read_csv(census_dat_file, header=0, comment='#')
    census_dat = census_dat[census_dat['COUNTY'] > 0]
    census_dat['fips'] = 0
#   generate fips from concatenation of STATE and COUNTY fields in census records
    for r in range(0, len(census_dat)):
        fips = '{:02d}{:03d}'.format(
            census_dat['STATE'].iloc[r], census_dat['COUNTY'].iloc[r])
    #   if (census_dat['county'].iloc[r] == 'Kalawao'):
    #       print('----------Kalawao---------',census_dat['population'].iloc[r])
    #       print(r,census_dat['COUNTY'].iloc[r],census_dat['STATE'].iloc[r],
    #               census_dat['county'].iloc[r],census_dat['state'].iloc[r], fips)
        census_dat['fips'].iloc[r] = int(fips)

#   aggregate populations of NYC borroughs into NY Times convention for
    nyc_counties = ('Queens', 'Richmond', 'Kings', 'Bronx', 'New York')
    nyc_c_filter = (census_dat['county'].isin(nyc_counties)
                    & census_dat['state'].isin(['New York']))
    nyc_population = int(census_dat[nyc_c_filter]['population'].sum())

#   remove nyc_counties from census data
    census_dat = census_dat[~nyc_c_filter]

#   create unique instances of fips,population combinations using set(..)
    county_state_pop = set(zip(census_dat['fips'], census_dat['population']))
    cs_pop = pd.DataFrame(county_state_pop, columns=('fips', 'population'))

#   get NYT data
    nyt_dat = pd.read_csv(cv.NYT_counties, header=0)
    nyt_dat = nyt_dat.sort_values(by=['fips'], ascending=False)
#   remove counties without fips code (mainly 'Unknown' counties
    empty_fips_filter = pd.notna(nyt_dat['fips'])
    nyt_dat = nyt_dat[empty_fips_filter]

#   create unique instances of NYT county & state combinations
    county_state_nyt = set(
        zip(nyt_dat['county'], nyt_dat['state'], nyt_dat['fips']))
    nyt_counties = pd.DataFrame(
        county_state_nyt, columns=('county', 'state', 'fips'))
    nyt_counties['code'] = None
    nyt_counties['flag'] = ' '
    nyt_counties = nyt_counties.sort_values(by=['fips'], ascending=False)

#   insert state postal codes and other abbreviation
    gcodes = pd.read_csv(cv.cv_home+'geography_codes.csv',
                         header=0, comment='#')
    for i in range(0, len(gcodes)):
        geog = gcodes['geography'].iloc[i]
        code = gcodes['code'].iloc[i]
        gfilter = nyt_counties['state'].isin([geog])
    #   avoid 'SettingWithCopyWarning':
        nyt_counties.loc[gfilter, 'code'] = code

#   merge the two data frames using NYT county designations
    nyt_census = nyt_counties.merge(right=cs_pop)
#   append row for New York City population
    nyc_row = pd.Series(['New York City', 'New York', 36999,
                        'NY', ' ', nyc_population], index=nyt_census.columns)
    nyt_census = nyt_census.append(nyc_row, ignore_index=True)
#   df.astype({'col1': 'int32'}).dtype
    nyt_census = nyt_census.astype({'fips': 'int64'})
    nyt_census = nyt_census.sort_values(
        by=['population', 'state', 'county'], ascending=False)

    print('nyt_census (Los Angeles, CA through Kenedy, TX):')
    print(nyt_census)
    nyt_census.to_csv('nyt_census.csv', index=False)

# ---------------- global utility functions ---------------------------


def Strptime(x):
    """
    wrapper for datetime.strptime callable by map(..)
    """
    y = datetime.strptime(x, '%Y-%m-%d')
    return(y)


def SD_lim(x, mult):
    M = statistics.mean(x)
    S = statistics.stdev(x)
    if (S > 0.0):
        multS = mult*S
        return([M-multS, M+multS])
    else:
        return(min(x), max(x))


def isNaN(num):
    return num != num


"""
def median(x):
    mx = x.quantile(q=0.5)
    return float(mx)
"""


def make_SD_tab(Gfile='top30.csv', save=True):

    def robust_sd(a):
        mask = [~np.isnan(a) & ~np.isinf(a)]
        sd = statistics.stdev(a[mask])
        return(sd)

    print('Reading:', cv.cv_home+Gfile)
    # , encoding = "ISO-8859-3")
    gg = pd.read_csv(cv.cv_home+Gfile, header=0, comment='#')
    print('Finished reading:', cv.cv_home+Gfile)
    print(gg.columns)

    SD_columns = ['County', 'Cases', 'Deaths', 'log Cases', 'log Deaths']
    SD_tab = pd.DataFrame(columns=SD_columns, dtype=None)
    row = pd.Series(index=SD_columns)

    print('Processing', len(gg), 'geographies')
    for g in range(0, len(gg)):
        print(g, gg['name'][g])
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

        SD_tab = SD_tab.append(row, ignore_index=True)

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
    SD_tab = SD_tab.append(row, ignore_index=True)
    print(SD_tab)


def plot_dow_boxes(nG=5):
    cv.population_dat = pd.read_csv(cv.GeoIndexPath, header=0, comment='#')
    gg = cv.population_dat

    dow_names = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
    week = [0, 1, 2, 3, 4, 5, 6]
    if nG > 5:
        rows = 2
        cols = 1
        ht = 2.25
        Cweight = 100000
        Dweight = 10000
    else:
        rows = nG
        cols = 2
        ht = 9.0/rows

    fig, ax = plt.subplots(rows, cols, figsize=(6.5, ht*rows))
    ecol = 'black'  # '#008fd5' # 'mediumblue'
    fcol = '#008fd5'  # 'cornflowerblue'

    allCcount = pd.DataFrame(columns=week)
    allDcount = pd.DataFrame(columns=week)
    population = pd.Series([]*nG)

    for g in range(0, nG):
        #   print(gg['county'].iloc[g])
        tmpG = Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                         code=gg['code'].iloc[g])
        tmpG.read_nyt_data('county')
        Ccount = pd.Series([0.0]*len(dow_names))  # ,index=dow_names)
        Dcount = pd.Series([0.0]*len(dow_names))  # ,index=dow_names)
        d1_cases = np.diff(tmpG.cases)
        d1_deaths = np.diff(tmpG.deaths)
        for k in range(0, len(tmpG.date)-1):
            j = datetime.strptime(tmpG.date[k], '%Y-%m-%d').weekday()
            Ccount[j] += d1_cases[k]
            Dcount[j] += d1_deaths[k]

        population[g] = tmpG.population
    #   print(g,tmpG.moniker,tmpG.population,Ccount.sum(),Dcount.sum())
    #   counties with zero reported deaths cause NaNs
        Ccount = Ccount/(Ccount.sum()+eps)
        Dcount = Dcount/(Dcount.sum()+eps)
        allCcount = allCcount.append(Ccount, ignore_index=True)
        allDcount = allDcount.append(Dcount, ignore_index=True)
    #   verify final day of week is correct with respect to calander
    #   k  = len(tmpG.date)-1
    #   print(tmpG.date[k],j,dow_names[j])

        if nG <= 5:
            ax[g, 0].bar(week, Ccount, tick_label=dow_names,
                         color=fcol, edgecolor=ecol)
            ax[g, 0].set_ylabel('Cases')
            ax[g, 1].bar(week, Dcount, tick_label=dow_names,
                         color=fcol, edgecolor=ecol)
            ax[g, 1].set_ylabel('Deaths')
            tx = prop_scale(ax[g, 0].get_xlim(), 0.1)
            ty = prop_scale(ax[g, 0].get_ylim(), 1.0)
            ax[g, 0].text(tx, ty, tmpG.moniker, ha='center', fontsize=8)

    if nG > 5:
        #   for g in range(0,nG):
        #       allCcount.iloc[g] = (Cweight/population[g])*allCcount.iloc[g]
        #       allDcount.iloc[g] = (Dweight/population[g])*allDcount.iloc[g]
        title = str(nG)+' most populous US counties'
        fig.text(0.5, 0.9, title, ha='center', va='bottom')
        ax[0].boxplot(allCcount.transpose(), labels=dow_names)
        ax[0].set_ylabel('Cases')  # per '+str(int(Cweight/1000))+'K')
        ax[1].boxplot(allDcount.transpose(), labels=dow_names)
        ax[1].set_ylabel('Deaths')  # per '+str(int(Dweight/1000))+'K')
        ax[1].set_ylim(0.0, 1.0)  # ax[0].get_ylim())

    gfile = cv.graphics_path+'days_of_week_'+str(nG)+'.png'
    plt.savefig(gfile, dpi=300)
    plt.show()


def web_update(years=['2020', '2021', '2022']):
    os.system('git -C /home/other/nytimes-covid-19-data pull -v')
    cv.nyt_county_dat.iloc[0:0]

    file = cv.NYT_home + '/us-counties-' + years[0] + '.csv'
    cv.nyt_county_dat = pd.read_csv(file, header=0)
    columns = cv.nyt_county_dat.columns
    cv.nyt_county_dat.columns = [''] * len(cv.nyt_county_dat.columns)
    print(file)
#    print(cv.nyt_county_dat)

    for k in range(1, len(years)):
        file = cv.NYT_home + '/us-counties-' + years[k] + '.csv'
        print(k, file)
        ydat = pd.read_csv(file, header=0)
        ydat.columns = [''] * len(ydat.columns)
#        print(ydat)
        cv.nyt_county_dat = pd.concat(
            [cv.nyt_county_dat, ydat], ignore_index=True)

    cv.nyt_county_dat.columns = columns
    print('Updated NYT data for years', years)
    print(cv.nyt_county_dat)
    cv.nyt_county_dat.to_csv(cv.NYT_counties, header=True, index=False)

#   print()
#   if 1: sys.exit(1)

    BC_cases_file = 'BCCDC_COVID19_Dashboard_Case_Details.csv'
#               http://www.bccdc.ca/Health-Info-Site/Documents/
#                    BCCDC_COVID19_Dashboard_Case_Details.csv
    cmd = 'wget --verbose http://www.bccdc.ca/Health-Info-Site/Documents/' + BC_cases_file +\
        ' -O '+cv.cv_home+BC_cases_file
    print(cmd)
    os.system(cmd)
    print('Updated BC cases')
    print()

    print('Updating CDC vax data')
    VV.get_cdc_dat(True)
    print('Updated  CDC vax data')
    print()


def make_dat_files():
    # pd.read_csv(cv.GeoIndexPath,header=0,comment='#')
    nyt_counties = cv.GeoIndex
    gg_filter = nyt_counties['flag'].str.contains('m')
    gg = nyt_counties[gg_filter]
    print(gg)

    for g in range(0, len(gg)):
        print(gg['county'].iloc[g])
        tmpG = GG.Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                            code=gg['code'].iloc[g])
        tmpG.read_nyt_data('county')
        tmpG.read_vax_data()
        tmpG.write_dat_file()
     #  tmpG.plot_prevalence(save=True,cumulative=False, show_order_date=False,
     #      show_superspreader=False,per_capita=True,nax = 4)


def update_fits(njob=4):

    make_dat_files()

    save_wd = os.getcwd()
    print('save:', save_wd)
    print(cv.TMB_path)
    os.chdir(cv.TMB_path)
    print('current', os.getcwd())
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
    print('Starting', cmd)
    os.system(cmd)
    print('Finished', cmd)
    os.chdir(save_wd)
    print('current', os.getcwd())


def make_prevalence_plots(flags=['m']):
    nyt_counties = cv.GeoIndex
    # get rid of blank flag fields
    nyt_counties = nyt_counties[nyt_counties['flag'] != ' ']

#   quartiles = GG.plot_prevalence_comp_histo(flag='500000',window=7,save=True,
#                                             signature=True)
#   print('prevalence quartiles:',type(quartiles))
#   print(quartiles)
#   print('q[0.05] =',quartiles[0.05])

    qq = GG.qcomp(flag='250000', window=7)
    print('prevalence quartiles:', type(qq))
    print(qq)

    GG.plot_prevalence_comp_TS(flag='L', save=True, signature=True, window=7,
                               qq=qq, pp=[0.2, 0.2, 0.2, 0.8])
    GG.plot_prevalence_comp_TS(flag='B', save=True, signature=True, window=7, qq=qq,
                               pp=[0.2, 0.2, 0.2, 0.8])
    print('Finished prevealence comp_TS plots')

    gg_filter = pd.Series(index=nyt_counties.index, dtype=bool)
    # 'or' the flags with the flag field
    for f in flags:
        gg_filter = gg_filter | nyt_counties['flag'].str.contains(f)

    gg = nyt_counties[gg_filter]

    for g in range(0, len(gg)):
        print(gg['county'].iloc[g])
        tmpG = GG.Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                            code=gg['code'].iloc[g])
        if (gg['code'].iloc[g] == 'BC'):
            tmpG.read_BCHA_data('province')
        else:
            tmpG.read_nyt_data('county')
            tmpG.read_vax_data()

        tmpG.plot_prevalence(save=True, signature=True, cumulative=False,
                             show_order_date=False, per_capita=True, window=[7],
                             qq=qq, pp=[0.2, 0.2, 0.2, 0.8])


def update_shared_plots():

    shared_plots = ['AlamedaCA_prevalence.png', 'District_of_ColumbiaDC_prevalence.png',
                    'FairfaxVA_prevalence.png', 'HonoluluHI_prevalence.png',
                    'Los_AngelesCA_prevalence.png', 'MarinCA_prevalence.png',
                    'MendocinoCA_prevalence.png', 'MultnomahOR_prevalence.png',
                    'OtsegoNY_prevalence.png', 'PinellasFL_prevalence.png',
                    'PlumasCA_prevalence.png', 'San_DiegoCA_prevalence.png',
                    'San_FranciscoCA_prevalence.png', 'Santa_ClaraCA_prevalence.png',
                    'SonomaCA_prevalence.png', 'TompkinsNY_prevalence.png',
                    'Vancouver_IslandBC_prevalence.png']
    # 'days_of_week_1000.png','days_of_week_5.png','prevalence_comp_TS_m.png',
    shared_path = cv.cv_home+'PlotsToShare/'
    for file in shared_plots:
        cmd = 'cp -pvf ' + cv.graphics_path+file+' ' + shared_path
        os.system(cmd)


def update_assets():
    asset_files = ['prevalence_comp_TS_B.png', 'prevalence_comp_TS_L.png', 'recent_prevalence_histo_pop.png',
                   'New_York_CityNY_prevalence.png', 'Los_AngelesCA_prevalence.png',
                   'CFR_all_5.png', 'CFR_all_0000.png', 'CFR_hist_all_recent.png',
                   'CFRridge_30_23.png']

    for file in asset_files:
        cmd = 'cp -pvf '+cv.graphics_path+file+' '+cv.assets_path
        os.system(cmd)


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
    cmd = 'cp -fv '+index_md + ' ' + index_md + '.bak'
#   print(cmd)
    os.system(cmd)

    cmd = "sed -e '/<!---START TABLE--->/,/<!---END TABLE--->/!b' "
#   print(cmd)
    cmd = cmd + "-e '/<!---END TABLE--->/!d;r"+table+"' "
#   print(cmd)
    cmd = cmd + "-e 'd' "+index_md+" > "+tmp
#   print(cmd)
    os.system(cmd)

    cmd = 'cp -fv '+tmp + ' ' + index_md
#   print(cmd)
    os.system(cmd)
#   print('Finished update html table in index.md')


def update_everything(do_fits=True, do_web=True):
    if do_web:
        web_update()
        print('Finished web_update ...')
    else:
        print('Omitting web_update ...')

    make_prevalence_plots(['L', 'm', 's'])
    print("Finished make_prevalence_plots(['L','m','s'])")
    update_shared_plots()
    print('Finished update_shared_plots()')
    CFR.plot_DC_scatter(save=True)
    CFR.plot_recent_CFR(save=True)
    CFR.CFR_comp(nG=30, w=23)
    CFR.plot_CFR_ridge('CFR_ridge.csv')
    print('Finished CFR plots')

    update_assets()
    print('Finished update asset directory')

    update_html()
    print('Finished update html table in index.md')

    if (do_fits):
        os.system('rm -v ' + cv.dat_path + '*.dat')
        make_dat_files()
        print('Finished make_dat_files()')

        update_fits()
        print('Finished update_fits()')
        FF.make_fit_plots()
        print('Finished fit_plots')
        FF.make_fit_table()
        print('Finished fit table')
        FF.make_rate_plots('logbeta', show_doubling_time=True, save=True)
        FF.make_rate_plots('logbeta', show_doubling_time=True, save=True,
                           fit_files=['Los_AngelesCA', 'New_York_CityNY'])
    #                   fit_files=['Miami-DadeFL','HonoluluHI','NassauNY','CookIL'])
        FF.make_rate_plots('logmu', save=True)
        print('Finished rate_plots')

    print('Finished Everything!')


def log_norm_cfr():
    def vline(ax, y, ylim, mark):
        ax.plot((y, y), ylim)
        GU.mark_ends(ax, (y, y), ylim, mark, 'r')

    import math
    import scipy.stats as stats
    cfr = np.array(pd.read_csv('recent_cfr.csv')['cfr'])

    bins = np.linspace(0.0, 0.1, 50)
    nbin = len(bins)

    fig, ax = plt.subplots(2, figsize=(6.5, 6.5))

    print('1 ----------------')
    weights = np.ones_like(cfr) / len(cfr)
    hist, edges, patches = ax[0].hist(cfr, nbin, weights=weights, density=True)

    print('hist sum:', sum(hist))

    width = max(edges)/nbin
    ax[0].set_xlabel('CFR')
    ax[0].set_ylabel('Number')

    params = stats.lognorm.fit(cfr)
    tedges = np.linspace(0.0, 0.1, 200)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    print('params:', params)
    print(loc, scale)
    prob = stats.lognorm.pdf(tedges, params[0], params[1], params[2])
    print('prob sum:', sum(prob))
    halfwidth = 0.5*(tedges[1]-tedges[0])
    ax[0].plot(tedges-halfwidth, prob, linewidth=2, linestyle='--')

    wparams = stats.weibull_min.fit(cfr)
    print('wparams:', wparams)
    wshape = wparams[0]
    wscale = params[2]
    print('wshape:', wshape)
    print('wscale:', wscale)
    wprob = stats.weibull_min.pdf(tedges, wparams[0], wparams[1], wparams[2])
    print('wprob sum:', sum(wprob))
    ax[0].plot(tedges-halfwidth, wprob, linewidth=3, linestyle=':')

    print('2 ----------------')
    logcfr = np.log(cfr)
    lhist, ledges = np.histogram(logcfr, bins=nbin, density=False)
    lwidth = 1.05*max(ledges)/nbin
    ax[1].bar(ledges[:-1], lhist, width=lwidth,
              color='0.85')  # next(prop_iter)['color']
    ax[1].set_xlabel('log CFR')
    ax[1].set_ylabel('Number')

    lparam = stats.norm.fit(logcfr)
    mean = np.exp(np.mean(logcfr))
    std = np.exp(np.std(logcfr))
    median = np.exp(np.median(logcfr))
    mode = tedges[pd.Series(prob).idxmax()]-halfwidth
    Q95 = np.quantile(cfr, q=0.95)
    Q99 = np.quantile(cfr, q=0.99)
    vlim = ax[0].get_ylim()
    vline(ax[0], mean, vlim, 'mean')
    vline(ax[0], mode, vlim, 'mode')
    vlim = (vlim[0], 0.8*vlim[1])
    vline(ax[0], Q95, vlim, '95%')
    vline(ax[0], Q99, vlim, '99%')

    lpdf = stats.norm.pdf(ledges, lparam[0], lparam[1])  # ,lparam[2])
#   ensure the area under the pdf agrees with the area under the bars
    lprob = len(logcfr)*lpdf/sum(lpdf)
    # ,color=next(prop_iter)['color']) #color='0.7', #,color=
    ax[1].plot(ledges, lprob)

    plt.savefig('log_norm_cfr.png', format='png', dpi=300)
    plt.show()


def git_commit_push():
    os.system(
        'git commit ~/Projects/SIR-Models/PlotsToShare/\*.png -m "Update PlotsToShare"')
    os.system('git commit ~/Projects/SIR-Models/assets/\*.png -m "Update assets"')
    os.system('git push')

    cmd = 'git -C ' + cv.Jon_path + " commit index.md -m 'update table'"
    print(cmd)
    os.system(cmd)
    cmd = 'git -C ' + cv.Jon_path + ' push'
    print(cmd)
    os.system(cmd)


def fit_lnCFR(CFRfile, Floc=None):
    CFR = pd.read_csv(CFRfile, header=0, index_col=0)
#   print(CFR.shape)
#   print(len(CFR.index))
#   print(len(CFR.columns))
#   print(CFR)
    nG = len(CFR.columns)
    ndate = len(CFR.index)

    # CFR log-normal parameters by date
    CFRln = pd.DataFrame(columns=('shape', 'loc', 'scale',
                                  'meanR', 'sdR', 'meanlogR', 'sdlogR'),
                         index=CFR.index)
    for date in CFR.index:
        if Floc is None:
            shape, loc, scale = stats.lognorm.fit(CFR.loc[date])
        else:
            shape, loc, scale = stats.lognorm.fit(CFR.loc[date], floc=Floc)
        meanR = np.mean(CFR.loc[date])
        sdR = np.std(CFR.loc[date])
        logR = np.log(CFR.loc[date])
        meanlogR = np.exp(np.mean(logR))
        sdlogR = np.exp(np.std(logR))
        CFRln.loc[date] = [shape, loc, scale, meanR, sdR, meanlogR, sdlogR]

    print(CFRln)

    file_name = 'CFRstats_{}_{}.csv'.format(nG, Floc)
    csv = open(file_name, 'w')
    csv.write('{} {}\n'.format(nG, Floc))
    CFRln.to_csv(csv, index=True)
    print('wrote CFNln to', file_name)


def plot_CFR_lines(CFRln_file):
    csv = open(CFRln_file)
    nG, floc = csv.readline().split(' ', 1)
    CFRdesc = pd.read_csv(csv, header=0, index_col=0)
    print('floc:', floc)
    print(CFRdesc)

    fig, ax = plt.subplots(1, figsize=(6.5, 4.5))
    bins = np.linspace(0.0, 0.1, 100)
    ax.set_xlabel('Case Fatality Ratio')
    ax.set_ylabel('Proportion')
    for d in range(0, len(CFRdesc)):
        #   for d in range(len(CFRdesc)-1,41,-1):
        #   for d in range(110,80,-1):
        pdf = stats.lognorm.pdf(
            bins, CFRdesc['shape'][d], CFRdesc['loc'][d], CFRdesc['scale'][d])
        pdf = pdf/sum(pdf)
        ax.plot(bins, pdf, linewidth=1)
        GU.mark_peak(ax, bins, pdf, str(d))

    tx = GU.prop_scale(ax.get_xlim(), 0.95)
    ty = GU.prop_scale(ax.get_ylim(), 0.8)
    txt = '{} counties, floc = {}\n'.format(nG, floc)
    ax.text(tx, ty, txt, ha='right', fontsize=10)
    plt.show(block=False)

    fig, ax = plt.subplots(1, figsize=(6.5, 4.5))
#   ax.set_xlabel('Date')
    ax.set_ylabel('Mean CFR')
    Date = mdates.date2num(CFRdesc.index)
    GU.make_date_axis(ax)
    ax.plot(Date, CFRdesc['meanR'])
    GU.mark_ends(ax, Date, CFRdesc['meanR'], 'arithmetic', 'r')
    ax.plot(Date, CFRdesc['meanlogR'])
    GU.mark_ends(ax, Date, CFRdesc['meanlogR'], 'log trans', 'r')

    plt.show()


def plot_CFR_contour(CFRln_file):
    csv = open(CFRln_file)
    nG, floc = csv.readline().split(' ', 1)
    CFRdesc = pd.read_csv(csv, header=0, index_col=0)
    CFRdesc.index = pd.DatetimeIndex(CFRdesc.index, freq='D')
    d1 = '2020-02-01'  # 00:00:00'
    d2 = cv.nyt_county_dat['date'].iloc[-1]
    print(d1, d2)
    date_list = pd.DatetimeIndex(pd.date_range(d1, d2), freq='D')

    bins = np.linspace(0.0, 0.10, 100)
    X = date_list
    Y = bins
    Z = pd.DataFrame(columns=bins, index=date_list)
    for d in date_list:
        pdf = stats.lognorm.pdf(bins, CFRdesc.loc[d]['shape'],
                                CFRdesc.loc[d]['loc'], CFRdesc.loc[d]['scale'])
        pdf = pdf/sum(pdf)
        pdf = pd.Series(pdf, index=bins)
        Z.loc[d] = pdf

    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))

    CS = ax.contour(X, Y, Z.transpose(), levels=np.linspace(
        0.0, 0.3, 100), linewidths=1)
    ax.clabel(CS, inline=1, fontsize=8)

    plt.show()


'''
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
'''


def plot_prevalence_stats_TS(flag=None, per_capita=True, mult=10000, delta_ts=True,
                             window=7, plot_dt=False, cumulative=False,
                             show_order_date=False,
                             show_superspreader=False,
                             annotation=True, signature=False,
                             ymaxdefault=None,
                             show_SE=False,
                             low_prev=1.0,
                             #                   ymax = [None,None,None], #[0.2,0.01,0.04],
                             save=True, nax=4):
    """ 

    """
    firstDate = '2020-03-01'  # mdates.date2num(cv.FirstNYTDate)
    mtime = os.path.getmtime(cv.NYT_home+'us-counties.csv')
#   dtime = datetime.fromtimestamp(mtime)
    lastDate = datetime.fromtimestamp(mtime)
#   print('GG lastDate:',lastDate)
#    orderDate = mdates.date2num(cv.CAOrderDate)
    print('GG firstDate,lastDate:', firstDate, lastDate)

    #dates = np.arange(firstDate,lastDate,1.0)
    # print(dates)
    cdate = pd.Series(np.arange(np.datetime64(firstDate),
                                np.datetime64(
                                    lastDate), np.timedelta64(1, 'D'),
                                dtype='datetime64[D]'))
    print(cdate)
    pdate = pd.Series(mdates.date2num(cdate), dtype=np.int64)
    # print(pdate)
    # if 1:
    #    sys.exit(1)

    cases = pd.DataFrame(index=pdate, dtype='object')
    cases['cdate'] = cdate
    print(cases)

    if 1:
        sys.exit(1)

    nax = 2  # 3
    fig, ax = plt.subplots(nax, 1, figsize=(6.5, nax*2.25))
    plt.rcParams['lines.linewidth'] = 1.5

    ylabel = ['Daily Cases', 'Daily Deaths', 'Case Fatality Ratio']
    if (per_capita):
        for a in range(0, 2):
            ylabel[a] = ylabel[a] + '/'+str(mult)
    total_names = ['Cases', 'Deaths', '']
    save_path = cv.graphics_path

    nyt_counties = pd.read_csv(cv.GeoIndexPath, header=0, comment='#')

    if flag.isnumeric():
        gg_filter = nyt_counties['population'] > float(flag)
    else:
        gg_filter = nyt_counties['flag'].str.contains(flag)
    gg = nyt_counties[gg_filter]

    if (ymaxdefault is None):
        ymax = [0.0]*3
    else:
        ymax = ymaxdefault

    nG = len(gg)

#   EndOfTime = dtime.date()+timedelta(days=21)
    #oldEndOfTime = cv.EndOfTime
    cv.EndOfTime = cv.dtime.date()+timedelta(days=7)

    for g in range(0, nG):
        print(g, gg['county'].iloc[g], gg['code'].iloc[g], gg['fips'].iloc[g])
        tmpG = GG.Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                            code=gg['code'].iloc[g])
        if (gg['code'].iloc[g] == 'BC'):
            tmpG.read_BCHA_data('province')
        else:
            tmpG.read_nyt_data('county')


#######################################################################
#######################################################################


# CFR.plot_CFR_lognorm_fit(save=False)
# fit_lnCFR('CFR1000.csv',Floc=0.0)
# plot_CFR_lines('CFRstats_1000_0.0.csv')
# plot_CFR_contour('CFRstats_1000_0.0.csv')
#tgeog = GG.Geography(name='Santa Clara',enclosed_by='California',code='CA')
#tgeog = GG.Geography(name='Harris',enclosed_by='Texas',code='TX')
#tgeog = GG.Geography(name='Hennepin',enclosed_by='Minnesota',code='MN')
#tgeog = GG.Geography(name='Los Angeles',enclosed_by='California',code='CA')
#tgeog = GG.Geography(name='New York City',enclosed_by='New York',code='NY')
#tgeog = GG.Geography(name='Pinellas',enclosed_by='Florida',code='FL')
#tgeog = GG.Geography(name='Honolulu',enclosed_by='Hawaii',code='HI')
#tgeog = GG.Geography(name='Pinellas',enclosed_by='Florida',code='FL')
#tgeog = GG.Geography(name='San Diego',enclosed_by='California',code='CA')
#tgeog = GG.Geography(name='Hamilton',enclosed_by='Ohio',code='OH')

'''
#tgeog = GG.Geography(name='Alameda', enclosed_by='California', code='CA')
tgeog = GG.Geography(name='San Joaquin', enclosed_by='California', code='CA')
tgeog.read_nyt_data('county')
tgeog.read_vax_data()
tgeog.print_metadata()
# tgeog.print_data()
qq = GG.qcomp(flag='1500000', window=7)
tgeog.plot_prevalence(save=True, signature=True, cumulative=False,
                      show_order_date=False, per_capita=True, window=[7],
                      qq=qq, pp=[0.2, 0.2, 0.2, 0.8], nax=4)

qq = GG.qcomp(flag='1500000',window=7)
GG.plot_prevalence_comp_TS(flag='B',save=True, signature=True,qq=qq,
                               pp = [0.2,0.2,0.2,0.8])
'''
#tmpG = GG.Geography(name='Vancouver Island',enclosed_by='British Columbia',code='BC')
#tmpG.read_BCHA_data()
#tmpG.plot_prevalence(save=False,signature=True,cumulative=False,
                     #per_capita=True,show_order_date=False)

# VV.get_cdc_dat(update=True)
# web_update()
# make_dat_files()
# update_fits()
# FF.make_fit_plots()
# FF.make_rate_plots('logbeta',show_doubling_time = False, save=False,
#                  #show_order_date = False,
#                  fit_files=['Los_AngelesCA','New_York_CityNY'])
#                  fit_files=['BrowardFL', 'NassauNY', 'MiddlesexMA',
#                             'MaricopaAZ', 'New_York_CityNY',
#                             'SuffolkNY', 'Miami-DadeFL'])
# FF.make_fit_table(path='obs_error')
#FF.make_fit_table(model_name = 'rrSIR')

#tfit = FF.Fit(cv.fit_path+'AlamedaCA.RData')
# tfit.plot(save=True,logscale=True,show_doubling_time=True)
#FF.make_rate_plots('logbeta',show_doubling_time = True,save=True)
# FF.make_rate_plots('logbeta',show_doubling_time = True, save=True,
#                    fit_files=['Los_AngelesCA','New_York_CityNY'])
# FF.make_rate_plots('logmu',save=True)

#GG.plot_prevalence_comp_TS(flag='B',save=True, signature=False)

# make_prevalence_plots(['L','m','s'])
# make_prevalence_plots(['B'])
# update_shared_plots()
# CFR.plot_DC_scatter(save=True)
# CFR.plot_recent_CFR(save=True)]]]]]]]]]]\[]
#CFR.CFR_comp(nG=30, w = 23)
# CFR.plot_CFR_ridge('CFR_ridge.csv')

#update_everything(do_fits=False,do_web=True)
#git_commit_push()

#GG.plot_prevalence_comp_TS(flag='B',save=True, signature=False,nax=3)
#GG.plot_prevalence_comp_TS(flag='L',save=True, signature=False)
#GG.plot_prevalence_comp_TS(flag='H',save=True, signature=True)
#GG.plot_prevalence_comp_TS(flag='m',save=True, signature=True)
#GG.plot_prevalence_comp_TS(flag='500000',save=True, signature=True)
### GG.plot_prevalence_comp_histo(flag='500000',window=15,save=True, signature=True)
# CFR.plot_recent_CFR(save=True)
# CFR.plot_DC_scatter(save=True)
# CFR.plot_recent_CFR(save=True)
# make_prevalence_plots(['L','m','s'])
#CFR.CFR_comp(nG=30, w = 23)
# CFR.plot_CFR_ridge('CFR_ridge.csv')

#qq=GG.plot_prevalence_comp_histo(flag='500000',window=15,save=False, signature=True)
# print('type(qq),qq:')
# print(type(qq),qq)
#GG.plot_prevalence_comp_TS(flag='m',low_prev=qq[0.05],save=False, signature=True)

# `qq = GG.qcomp(flag='250000', window=7, signature=False)

#qq = GG.qcomp(flag='1000000', window=7, signature=False)
# print(qq)
# print(qq.shape)
# print(qq['cases'][0.1])


#
# GG.plot_prevalence_comp_TS(flag='B',save=False, signature=True,window=7,
#                              qq = qq, pp = [0.2,0.2,0.2,0.8])

#GG.plot_prevalence_comp_histo(flag='950000',window=15,save=False, signature=True, qq = qq, p = 0.2)
#BC_cases_file = 'BCCDC_COVID19_Dashboard_Case_Details.csv'
#               http://www.bccdc.ca/Health-Info-Site/Documents/
#                    BCCDC_COVID19_Dashboard_Case_Details.csv
# cmd = 'wget --verbose http://www.bccdc.ca/Health-Info-Site/Documents/' + BC_cases_file +\
#' -O /home/jsibert/Desktop/'+BC_cases_file
# print(cmd)
# os.system(cmd)
# update_html()
