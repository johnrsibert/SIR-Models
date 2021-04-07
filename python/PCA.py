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

#def cov_comp(df):
#    """
#    expects rows with NaNs to be removed
#    compute covariance matrix from normalized data
#    """
#    cov = df.transpose().dot()
#    return(cov)
    
def Z_comp(df):
    z = pd.DataFrame(index=df.index,columns=df.columns)
    for c in df.columns:
        z[c] = (df[c] - df[c].mean())/df[c].std()
    return(z)    
    
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
    ggcode  = ['XX']*nG

    for g in range(0,nG):#len(gg)):
        tmpG = GG.Geography(name=gg['county'].iloc[g], enclosed_by=gg['state'].iloc[g],
                            code=gg['code'].iloc[g])
        print(g,gg['county'].iloc[g],gg['code'].iloc[g])
        ggcode[g] = gg['code'].iloc[g]
        if (gg['code'].iloc[g] == 'BC'):
            tmpG.read_BCHA_data()
        else:
            tmpG.read_nyt_data('county')
 
        date = pd.Series(tmpG.get_pdate()[1:])
        cases  =  np.diff(tmpG.cases)/tmpG.population

        if (tmpG.deaths is not None):
            deaths = np.diff(tmpG.deaths)/tmpG.population
        else:
            deaths = [None]*len(cases)  #pd.Series(data=[None]*len(date),index = date)
        #    print('deaths:',deaths)
        #    return(np.NaN)
        if (which=='Cases'):
            dat = pd.Series(data=cases, index = date)
        elif (which == 'Deaths'):
            dat = pd.Series(data=deaths,index = date)
        else:
            print('which =',which,'not currently supported')
            return(2)
                
        gdf[tmpG.moniker] = dat   

    
    print('gdf before drop:',gdf.shape)
#    print(gdf)
#   drop all columns with all NA        
    gdf = gdf.dropna(axis=1,how='all')
    print('gdf after drop 1:',gdf.shape)
#   drop all rows with aany NA       
    gdf = gdf.dropna(axis=0,how='any')
    print('gdf after drop 2:',gdf.shape)
#    print(gdf)

    gname = gdf.columns
    ggcolor = ['k']*len(gname)
    for k, g in enumerate(gname):
        if ggcode[k] in ['CA','AZ','NV','OR','WA','NM']:
            ggcolor[k] = colors[0]
        elif ggcode[k] in ['TX','FL','LA','AL']:
            ggcolor[k] = colors[1]
        elif ggcode[k] in ['NY','NJ','CT','MA','PA','RI','ME']:
            ggcolor[k] = colors[2]
        elif ggcode[k] in ['IL','MN','MI','ID','OH']:    
            ggcolor[k] = colors[3]
        else:
            ggcolor[k] = 'k'
#    print(stcode)

    for k, s in enumerate(season_dates):
        smask = gdf.index.isin(season_range[k])

    gZ = Z_comp(gdf)
#    gcov = cov_comp(gZ)
    gcov = gZ.transpose().dot(gZ)

    print('gcov:',gcov.shape)
#    print(gcov)
    
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
    gy = pd.DataFrame(gZ.dot(gvL),index=gdf.index)
    print('gy:',gy.shape)
 
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
        
    gax1.set_ylabel('Component Score', ha="center")

    show_save(plt,save=True,which=which,what='gdf')


##############################################################
    
    tdf= gdf.transpose()
    tZ = Z_comp(tdf)
    tcov = tZ.transpose().dot(tZ)
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

    tvL = tv[:,0:LL]
    print('tvL:',tvL.shape)
#    ty = tdf.transpose().dot(tvL)
    ty = pd.DataFrame(tZ.dot(tvL),index=tdf.index)
    print('ty:',ty.shape)
    
    nax = 1
    tfig, tax = plt.subplots(nax,1,figsize=(6.5,nax*6.5))
    tax.scatter(ty[0],ty[1],c=ggcolor, marker=',')#,s=9)
    tax.set_xlabel('C 1')
    tax.set_ylabel('C 2')

#   col = ['k']*len(stcode)
    for g in range(0,len(ggcode)):
        tax.text(ty.iloc[g,0], ty.iloc[g,1],ggcode[g], color=ggcolor[g],
                 ha='center',va='center',fontsize=10,alpha=0.5)

    show_save(plt,save=True,which=which,what='tdf')
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection='3d')
    ax3.scatter(ty[0],ty[1],ty[2],c=ggcolor,marker=',',depthshade=True)
    for g in range(0,len(ggcode)):
        ax3.text(ty.iloc[g,0],ty.iloc[g,1],ty.iloc[g,2],ggcode[g],
                 c=ggcolor[g],ha='center',va='center',fontsize=10) #,alpha=0.5)
    ax3.set_xlabel('C 1')
    ax3.set_ylabel('C 2')
    ax3.set_zlabel('C 3')
    plt.show()
    
    show_save(plt,save=True,which=which,what='tdf3d')   
    

    return('Finished PCA(...)')

 
print(PCA(flag='m')) #None,minpop=100000))#,which='Deaths')) #None,minpop=50000,which='Deaths'))   
