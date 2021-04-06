#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 06:57:45 2021

@author: jsibert
"""

from covid21 import config as cv
from covid21 import Geography as GG
from covid21 import GraphicsUtilities as GU
import numpy as np
import pandas as pd
#from datetime import date, datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from numpy import linalg as LA

def cov_comp(df):
    """
    expects rows with NaNs to be removed
    normalize
    compute covariance matrix from normalized data
    """
    z = pd.DataFrame(index=df.index,columns=df.columns)
    for c in df.columns:
        tmp = pd.Series(data=df[c],index = df.index)
#        z[c] = (df[c] - df[c].mean())/df[c].std()
        
        tmp = (tmp-tmp.mean())/tmp.std()
        z[c] = tmp


    cov = z.transpose().dot(z)
    return(cov)
    
def show_save(plt,save,which,what):
    if save:
#       gfile = cv.graphics_path+'PCA_'+what+'_'+str(n)+'.png'
        gfile = 'PCA_'+what+'_'+which+'.png'
        plt.savefig(gfile,dpi=300)
        plt.show(block=False)
        plt.pause(5)
        print('Plot saved as',gfile)
    else:
        plt.show()
    plt.close()
  
def PCA(flag='m', minpop=250000, which = 'Cases',
                    show_order_date = False, 
                    show_superspreader = False,
                    annotation = True, signature = False, 
                    save = True):
    firstDate = cv.FirstNYTDate
    lastDate  = cv.EndOfTime
#    orderDate = mdates.date2num(cv.CAOrderDate)
    date_index = pd.date_range(start=firstDate, end=lastDate)#, freq='D')
    date_index = mdates.date2num(date_index)
    season_dates = [datetime.strptime('2020-05-31','%Y-%m-%d'),
               datetime.strptime('2020-09-30','%Y-%m-%d'),
               datetime.strptime('2021-02-28','%Y-%m-%d'),
               lastDate]
    season_dates = mdates.date2num(season_dates)   
    season_names=('Spring 2020','Summer 2020','Winter 2020','Spring 2021')    
    season_range =[[]]*len(season_dates)
    fromd = mdates.date2num(firstDate)
    for k, s in enumerate(season_dates):
        print(k,season_names[k]+':',fromd,'to',s)
        season_range[k] = np.arange(fromd,s,1)
#        print(season_range[k])
        fromd = s
        
    save_path = './'
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    gdf = pd.DataFrame(index=date_index)
    ndx = pd.read_csv(cv.GeoIndexPath,header=0,comment='#')
    print('ndx:',ndx.shape)
    print(ndx)

    if (flag is None):
        pmask = ndx['population'] > minpop 
        gg=ndx[pmask]                
        print(gg.shape)
        
    else:
        gg_filter = ndx['flag'].str.contains(flag)
        gg = ndx[gg_filter]

    print('gg:')
    print(gg)
    nG = len(gg)
    for g in range(0,nG):#len(gg)):
        tmpG = GG.Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                            code=gg['code'].iloc[g])
#        tmpG = GG.Geography(name=gg['county'].iloc[g], 
#                            enclosed_by=gg['state'].iloc[g], 
#                            code=gg['code'].iloc[g])
        print(g,gg['county'].iloc[g],gg['code'].iloc[g])
        if (gg['code'].iloc[g] == 'BC'):
            tmpG.read_BCHA_data()
        else:
            tmpG.read_nyt_data('county')
 
#    print(g,cols[g])
        date = pd.Series(tmpG.get_pdate()[1:])
#    print(date)
        cases  =  np.diff(tmpG.cases)/tmpG.population
#       print(g,'cases:')
#       print(cases)

        if (tmpG.deaths is not None):
            deaths = np.diff(tmpG.deaths)/tmpG.population
        else:
            deaths = None

        if (which=='Cases'):
            dat = pd.Series(data=cases,index = date)
            #    print('normalized DCA:')
            #    print(dat)
        elif (which == 'Deaths'):
            dat = pd.Series(data=deaths,index = date)
        else:
            print('which =',which,'not currently supported')
            return(2)
                
#       gdf[cols[g]]=dat
        gdf[tmpG.moniker] = dat   

    
    gdf = gdf.dropna(axis=0,how='any')

    gname = gdf.columns
    stcode = ['XX']*len(gname)
#   print(stcode)
    for k, g in enumerate(gname):
#        print(k,g,len(g))
        stcode[k] = g[len(g)-2:len(g)]
#    print(stcode)

    for k, s in enumerate(season_dates):
        smask = gdf.index.isin(season_range[k])

    gcov = pd.DataFrame(cov_comp(gdf))
    print('gcov:',gcov.shape)
    
    gw, gv = LA.eig(gcov)
    gw = np.real(gw)
    gv = np.real(gv)
    ww = sum(gw)
    tprop = 0
    for k in range(0,len(gw)):
        prop=gw[k]/ww
        tprop = tprop+prop
        print('{0: 3d}{1: .3f}{2: .3f}'.format(k,prop,tprop))
        if (tprop > 0.9):
            break

    print('Sum:',ww)

    LL = 3
    gvL = gv[:,0:LL]
    gy = pd.DataFrame(gdf.dot(gvL),index=gdf.index)
    print('gy:',gy.shape)
 
#    nax = 2
#    gfig, gax = plt.subplots(nax,1,figsize=(6.5,nax*3.0))
    gfig = plt.figure(figsize=(6.5,9.0))
    gax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3,rowspan=1)
    gax2 = plt.subplot2grid((3, 3), (1, 0), colspan=3,rowspan=2)


    for k, s in enumerate(season_dates):
        smask = gdf.index.isin(season_range[k])
        gax2.scatter(gy.iloc[smask,0], gy.iloc[smask,1], s=9,
                      label=season_names[k])
        
    gax2.set_xlabel('C 1')
    gax2.set_ylabel('C 2')
    gax2.legend(loc='lower right', frameon=False, framealpha=0.5,
                 markerfirst=False, fontsize='x-small', markerscale=3)


    GU.make_date_axis(gax1)
    gdate = gdf.index.values
    for a in range(0,LL):
        gax1.plot(gdate,gy[a],linewidth=1.5)
#        print(a,str(a))
        GU.mark_ends(gax1,gdate,gy[a].values,str(a+1),'b')
        
    gax1.set_ylabel('Component', ha="center")

#    plt.show()#block=False)
#   def show_save(plt,save,which,what):

    show_save(plt,save=True,which=which,what='gdf')


##############################################################
    
#   tdf= gdf.transpose()
    tdf= gdf

    tcov = tdf.dot(tdf.transpose())
#   cov2 = gdf.dot(gdf.transpose())
    print('tcov:',tcov.shape)
#    print(tcov)
 
    tw, tv = LA.eig(tcov)
    tw=np.real(tw)
    tv=np.real(tv)
    ww = sum(tw)
    tprop = 0
    for k in range(0,len(tw)):
        prop=tw[k]/ww
        tprop = tprop+prop
        print('{0: 3d}{1: .3f}{2: .3f}'.format(k,prop,tprop))
        if (tprop > 0.9):
            break

    print('Sum:',ww)

#    LL = 3
    tvL = tv[:,0:LL]
    print('tvL:',tvL.shape)
    ty = tdf.transpose().dot(tvL)

    print('ty:',ty.shape)
    
    nax = 1
    tfig, tax = plt.subplots(nax,1,figsize=(6.5,nax*6.5))
    tax.scatter(ty[0],ty[1],alpha=0.5,s=5)
    tax.set_xlabel('C 1')
    tax.set_ylabel('C 2')

    for g in range(0,len(stcode)):
        if stcode[g] in ['CA','AZ','NV','OR','WA','NM']:
            col = colors[0]
        elif stcode[g] in ['TX','FL','LA']:
            col = colors[1]
        elif stcode[g] in ['NY','NY','CT','MA','PA','IL','MN','MI']:
            col = colors[2]
        else:
            col = 'k'
        tax.text(ty.iloc[g,0], ty.iloc[g,1],stcode[g], color=col,
                 ha='center',va='center',fontsize=10) #,alpha=0.5)

#    plt.show()#block=False)
    show_save(plt,save=True,which=which,what='tdf')

    if (1):
        return('Finished PCA(...)')

 
print(PCA(flag=None,minpop=100000,which='Deaths'))   
