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
import matplotlib.patches as mpatches
from datetime import date, datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from numpy import linalg as LA
import os

plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams["scatter.marker"] = '.' 
plt.rcParams["lines.markersize"] = 1


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
    
def show_save(plt,root,save=True):
    if save:
        gfile = cv.graphics_path+root+'.png'
    #   gfile = 'PCA_'+what+'_'+which+'.png'
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
#   lastDate  = cv.EndOfTime
    mtime = os.path.getmtime(cv.NYT_home+'us-counties.csv')
    dtime = datetime.fromtimestamp(mtime)
    lastDate = dtime.date() - timedelta(days=1)
 
#    orderDate = mdates.date2num(cv.CAOrderDate)
    date_index = pd.date_range(start=firstDate, end=lastDate)#, freq='D')
    print('date_index:',date_index)
    date_index = mdates.date2num(date_index)

    season_dates = [datetime.strptime('2020-05-31','%Y-%m-%d'),
               datetime.strptime('2020-09-30','%Y-%m-%d'),
               datetime.strptime('2021-02-28','%Y-%m-%d'),
               lastDate]
    season_dates = mdates.date2num(season_dates)   
    season_names=('Spring 2020','Summer 2020','Winter 2020','Spring 2021')    
    season_range =[[]]*len(season_dates)

    month_names = ('Jan-Feb','Mar-Apr','May-Jun','Jul-Aug','Sep-Oct','Nov-Dec')
    print(month_names)

    fromd = mdates.date2num(firstDate)
    for k, s in enumerate(season_dates):
        print(k,season_names[k]+':',fromd,'to',s)
        season_range[k] = np.arange(fromd,s,1)
#        print(season_range[k])
        fromd = s
        
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
    #    print(g,gg['county'].iloc[g],gg['code'].iloc[g],tmpG.population,tmpG.date0)
        ggcode[g] = gg['code'].iloc[g]
        if (gg['code'].iloc[g] == 'BC'):
            tmpG.read_BCHA_data()
        #    tmpG.print_metadata()
        #    tmpG.print_data()
        else:
            tmpG.read_nyt_data('county')
            
        print(g,gg['county'].iloc[g],gg['code'].iloc[g],tmpG.population,tmpG.date0)

        pdate = pd.Series(tmpG.get_pdate()[1:])
        cases  =  np.diff(tmpG.cases)/tmpG.population

        if (tmpG.deaths is not None):
            deaths = np.diff(tmpG.deaths)/tmpG.population
        else:
            deaths = [None]*len(cases)  #pd.Series(data=[None]*len(date),index = date)
        #    print('deaths:',deaths)
        #    return(np.NaN)
        if (which=='Cases'):
            dat = pd.Series(data=cases, index = pdate)
        elif (which == 'Deaths'):
            dat = pd.Series(data=deaths,index = pdate)
        else:
            print('which =',which,'not currently supported')
            return(2)
                
        gdf[tmpG.moniker] = dat   

    gdf.to_csv('gdf.csv')
    
    print('gdf before drop:',gdf.shape)
    print(gdf)
#   drop all columns with all NA        
    gdf = gdf.dropna(axis=1,how='all')
    print('gdf after drop column with all NA:',gdf.shape)
#   drop all rows with any NA       
    gdf = gdf.dropna(axis=0,how='any')
    print('gdf after drop drop with any NA:',gdf.shape)
    print(gdf)

    gdf_index = gdf.index
    mcolors = pd.Series(0,index=gdf_index)
    mlabels = pd.Series('  ',index=gdf_index)
    for g in gdf_index:
    #    print(d.strftime("%m"))
    # date_index = mdates.date2num(date_index)
         d = mdates.num2date(g)
         m = int(d.strftime("%m"))
         c = int((m+1)/2)-1
         mcolors[g] = c
         mlabels[g] = month_names[c]
         print(g,d, d.strftime("%m"),m,c,mcolors[g],mlabels[g])

    print(mcolors)
 #  if (1): return('Stopped HERE')


    gname = gdf.columns
    ggcolor = ['k']*len(gname)
    for k, g in enumerate(gname):
        if ggcode[k] in ['WA','OR','CA','AZ','NV','HI','AK']:#,'BC']:
            ggcolor[k] = colors[0]
        elif ggcode[k] in ['ID','UT','MT','WY','NM','CO']:
            ggcolor[k] = colors[1]
        elif ggcode[k] in ['ND','SD','NE','KS','OK']:
            ggcolor[k] = colors[2]
        elif ggcode[k] in ['MN','WI','IA','IL','MO','IN','OH','MI']:    
            ggcolor[k] = colors[3]
        elif ggcode[k] in ['PA','NJ','NY','CT','MA','ME','RI','NH','VT','DE',
                           'DC']:
            ggcolor[k] = colors[4]
        elif ggcode[k] in ['TX','LA','MS','AL','FL','SC','NC','VA','MD','GA',
                           'TN','WV','OK','KY','AR']:
            ggcolor[k] = colors[5]
        else:
            ggcolor[k] = 'k'
#    print(stcode)
            
    monpatches = [plt.Artist]*len(colors)
    rpatches = [plt.Artist]*len(colors)
    rlabels = ['West','Mountain','Prairie','Midwest','Northeast','South']
 #   print('patches:',patches)
    for k in range(0,len(colors)):
        rpatches[k]   = mpatches.Patch(color=colors[k],label=rlabels[k])
    #   monpatches[k] = mpatches.Patch(color='k',label=mlabels[k])
 
    for k, s in enumerate(season_dates):
        smask = gdf.index.isin(season_range[k])
        
    title = str(nG)+' places'
    if (flag is None):
        title = title + '; pop. > ' + str(minpop)
    title = title + '; '+which

    groot = 'PCA_'+which+'_'+str(nG)

    tprop = pd.DataFrame(0.0,index=np.arange(0,max(gdf.shape)),
                             columns=['gdf','cgdf','tdf','ctdf'])
    print(tprop.shape)

    gZ = Z_comp(gdf)
    gcov = gZ.transpose().dot(gZ)

    print('gcov:',gcov.shape)
    print(gcov)
    
    gw, gv = LA.eig(gcov)
    gw = np.real(gw)
    gv = np.real(gv)
    ww = sum(gw)
    tprop['cgdf'][0] = tprop['gdf'][0]
    for k in range(0,len(gw)):
        tprop['gdf'][k] = gw[k]/ww
        if (k>0):
            tprop['cgdf'][k] = tprop['cgdf'][k-1] + tprop['gdf'][k]
        else:
            tprop['cgdf'][k] = tprop['gdf'][k]

    LL = 3
    gvL = gv[:,0:LL]
    gy = pd.DataFrame(gZ.dot(gvL),index=gdf.index)
    print('gy:',gy.shape)
    print(gy)
 
    gfig = plt.figure(figsize=(6.5,9.0))
    gax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3,rowspan=1)
    gax2 = plt.subplot2grid((3, 3), (1, 0), colspan=3,rowspan=2)
    gfig.suptitle(title+' C(X)',y=1.0)

    for k, s in enumerate(season_dates):
        smask = gdf.index.isin(season_range[k])
        gax2.scatter(gy.iloc[smask,0], gy.iloc[smask,1], s=100,
                      label=season_names[k])
#   gax2.scatter(gy[0],gy[1],c=mcolors,marker='.',s=50)#s=11)
        
    gax2.set_xlabel('C 1')
    gax2.set_ylabel('C 2')
#   gax2.legend(handles=monpatches,frameon=False, framealpha=0.5,
#                markerfirst=False, fontsize='x-small', markerscale=3)
    gax2.legend(loc='lower right', frameon=False, framealpha=0.5,
                 markerfirst=False, fontsize='x-small', markerscale=3)
#   gax2.legend(frameon=False, framealpha=0.5, labels=mlabels,
#                markerfirst=False, fontsize='x-small', markerscale=3)
    if (annotation):
        GU.add_data_source(gfig, 'Multiple sources')
        GU.add_signature(gfig,'https://github.com/johnrsibert/SIR-Models/tree/master/PlotsToShare')



    GU.make_date_axis(gax1)
    gdate = gdf.index.values
    for a in range(0,LL):
        gax1.plot(gdate,gy[a],linewidth=1.5)
#        print(a,str(a))
        GU.mark_ends(gax1,gdate,gy[a].values,str(a+1),'b')
        
    gax1.set_ylabel('Component Score', ha="center")
    gax2.set_ylim(gax1.get_ylim())
    gax2.set_xlim(gax1.get_ylim())

    show_save(plt,groot+'_gdf')

##############################################################
    
    tdf= gdf.transpose()
    print('transposed gdf:',tdf.shape)
    tZ = Z_comp(tdf)
    tcov = tZ.transpose().dot(tZ)
    print('tcov:',tcov.shape)
    print(tcov)

#   print('isnan:')
#   print(np.isnan(tcov).any())
#   print('isinf:')
#   print(np.isinf(tcov).any())
    if(any(np.isinf(tcov).any())):
        tcov = np.nan_to_num(tcov)
#       print(np.isinf(tcov).any())

 
    tw, tv = LA.eig(tcov)
    tw=np.real(tw)
    tv=np.real(tv)
    ww = sum(tw)
    tprop['ctdf'][0] = tprop['tdf'][0]
    for k in range(0,len(tw)):
        tprop['tdf'][k] = tw[k]/ww
        if (k>0):
            tprop['ctdf'][k] = tprop['ctdf'][k-1] + tprop['tdf'][k]
        else:
            tprop['ctdf'][k] = tprop['tdf'][k]

    tprop.to_csv('var_prop_'+which+'_'+str(nG)+'.csv',index=False)

    tvL = tv[:,0:LL]
    print('tvL:',tvL.shape)
#    ty = tdf.transpose().dot(tvL)
    ty = pd.DataFrame(tZ.dot(tvL),index=tdf.index)
    print('ty:',ty.shape)
    print(ty)
    
    tfig, tax = plt.subplots(1,1,figsize=(6.5,6.5))
    tax.scatter(ty[0],ty[1],c=ggcolor, marker=',')#,s=9)
    tax.set_xlabel('C 1')
    tax.set_ylabel('C 2')
    tax.legend(handles=rpatches,frameon=False, framealpha=0.5,
                 markerfirst=False, fontsize='x-small', markerscale=3)
                 #loc='lower right', 
    tfig.suptitle(title+" C(X')",y=1.02)
    if (annotation):
        GU.add_data_source(tfig, 'Multiple sources')
        GU.add_signature(tfig,'https://github.com/johnrsibert/SIR-Models/tree/master/PlotsToShare')


#   col = ['k']*len(stcode)
    for g in range(0,len(ggcode)):
        tax.text(ty.iloc[g,0], ty.iloc[g,1],ggcode[g], color=ggcolor[g],
                 ha='center',va='center',fontsize=10,alpha=0.5)

    show_save(plt,groot+'_tdf')
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection='3d')
    ax3.scatter(ty[0],ty[1],ty[2],c=ggcolor,depthshade=True)
    for g in range(0,len(ggcode)):
        ax3.text(ty.iloc[g,0],ty.iloc[g,1],ty.iloc[g,2],ggcode[g],
                 c=ggcolor[g],ha='center',va='center',fontsize=10) #,alpha=0.5)
    ax3.set_xlabel('C 1')
    ax3.set_ylabel('C 2')
    ax3.set_zlabel('C 3')
    fig3.suptitle(title+" C(X')")
    if (annotation):
        GU.add_data_source(fig3, 'Multiple sources')
        GU.add_signature(fig3,'https://github.com/johnrsibert/SIR-Models/tree/master/PlotsToShare')

#   plt.show()
    
    show_save(plt,groot+'_tdf3d')

    return('Finished PCA(...)')


print(PCA(None,minpop=int(1e6))) #,which='Deaths')) #None,minpop=50000,which='Deaths'))   
