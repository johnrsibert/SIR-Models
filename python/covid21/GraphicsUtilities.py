from covid21 import config as cv

from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import math

def make_date_axis(ax, first_prev_date = None):
    if (first_prev_date is None):
        firstDate = mdates.date2num(cv.FirstNYTDate)
    else:
        firstDate = first_prev_date
        
    lastDate  = mdates.date2num(cv.EndOfTime)
#   print('GU firstDate,lastDate:',firstDate,lastDate)
    ax.set_xlim([firstDate,lastDate])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
#   ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval = 7))

def none_reported(ax,what):
    lim = ax.get_xlim()
    tx = lim[0] + 0.5*(lim[1]-lim[0])
    lim = ax.get_ylim()
    ty = 0.5*lim[1] #lim[0] + 0.5*(lim[1]-lim[0])
    note = 'No '+what+' Reported'
    print(note)
    ax.text(tx,ty,note,ha='center',va='center',fontstyle='italic')

def plot_dtslopes(ax, xvar, yvar, threshold = 1000, dt = [1,2,4,8]):
    """
    Superimpose exponential growth slope lines for different doubling times
    This function is only relevant to demonstrate potential exponential
    growth at the beginning of an outbreak/
    ax: axisis on which to draw slopes lines
    xvar = date range
    yvar = cumulative cases "usually"
    threshold: number of cases used to start slopes
    dt: representative doubling times in days
    """
    k0 = 0
    # find frist date in shich cases exceeds threshold
    for k in range(0, len(yvar)-1):
        if (yvar[k] >= threshold):
            k0 = k
            break

    x0 = xvar[k0]
    y0 = yvar[k0]
    sl = np.log(2.0)/dt
    xrange = ax.get_xlim()
    yrange = [25,ax.get_ylim()[1]]
    for i in range(0,len(dt)):
        y = y0 + np.exp(sl[i]*(x0-xrange[0]))
        ax.plot([x0,xrange[1]],[y0,y],color='black',linewidth=1)
        c = ax.get_lines()[-1].get_color()
        mark_ends(ax,xrange,[y0,y],str(dt[i])+' da','r')
    return(ax)

def mark_peak(ax,x,y,label):
    c = ax.get_lines()[-1].get_color()
    a = ax.get_lines()[-1].get_alpha()
    i = pd.Series(y).idxmax()
    if (i>0):
      mark = ax.text(x[i],y[i],label,ha='center',va='bottom',fontsize=8, 
                     color=c) #,alpha=a)
      mark.set_alpha(a) # not supported on all backends

def mark_ends(ax,x,y,label,end='b',spacer=' '):
    '''
    x,y pandas Series containing coordinates of the line
    '''
#   print('len(ax.get_lines())',len(ax.get_lines()))
    c = ax.get_lines()[-1].get_color()
    a = ax.get_lines()[-1].get_alpha()
    mark = None
#   print('color, alpha:',c,a)
    if ( (end =='l') | (end == 'b') ):
    #     if (math.isfinite(x[0]) and math.isfinite(y[0])):
        try:
            mark = ax.text(x.iloc[0],y.iloc[0],label+spacer,ha='right',va='center',
                           fontsize=8, color=c) #,alpha=a)
        except Exception as exception:
            print('Unable to mark left end',(x.iloc[0],y.iloc[0]), 'for', 
                  label,(exception.__class__.__name__))
                    
    if ( (end =='r') | (end == 'b') ):
        try:
            mark = ax.text(x.iloc[-1],y.iloc[-1],spacer+label,ha='left',va='center',
                           fontsize=8, color=c) #,alpha=a)
        except Exception as exception:
            print('Unable to mark right end',(x.iloc[-1],y.iloc[-1]), 'for', 
                  label,(exception.__class__.__name__))
        

    # Set the alpha value used for blending
    if mark is not None:
        mark.set_alpha(a) # not supported on all backends

def add_superspreader_events(Date,adc,ax):
    sslag = 14
    ssdates = ['2020-06-19','2020-07-04','2020-11-26','2020-12-25','2020-08-09','2021-01-01']
    ax.plot([], []) # advance color cycler
    c = ax.get_lines()[-1].get_color()
    for d in ssdates:
        d1 = mdates.date2num(datetime.strptime(d,'%Y-%m-%d').date())
        d2 = d1 + sslag
        i1 = Date.index[list(Date).index(d1)]
        try:
            i2 = Date.index[list(Date).index(d2)]
        except:
            i2 = len(adc)-1
        y1 = adc[i1]
        y2 = adc[i2]
        ax.plot((d1,d1,d2),(y1,y2,y2),color=c,
                linewidth=2) #,linestyle=(0,(1,1,))) #'dotted')
               

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
    sd_region = plt.Polygon(np.transpose(xy), alpha=0.2, facecolor=c,
                            edgecolor=c,lw=0.5)
#                           edgecolor='0.1',lw=0.5)
    ax.add_patch(sd_region)


def add_order_date(ax,linewidth=5,alpha=0.5):
#   Newsome's shelter in place order
#   orderDate = mdates.date2num(cv.CAOrderDate)
    ax.axvline(mdates.date2num(cv.CAOrderDate),color='black', 
              linewidth=linewidth,alpha=alpha)
    ax.axvline(mdates.date2num(cv.HalfMillionShotDate),color='green',
              linewidth=linewidth,alpha=alpha)
    ax.axvline(mdates.date2num(cv.DexamethasoneDate),color='blue',
              linewidth=linewidth,alpha=alpha)
    ax.axvline(mdates.date2num(cv.IndependenceDay),color='red', 
              linewidth=linewidth,alpha=alpha)

def add_data_source(fig,source='Multiple sources.'):
#   if source is None:
#       source = 'New York Times, https://github.com/nytimes/covid-19-data.git.'
    fig.text(0.0,0.0,' Data source: '+ source , ha='left',va='bottom', fontsize=8)
    mtime = os.path.getmtime(cv.NYT_home+'us-counties.csv')
    dtime = datetime.fromtimestamp(mtime)
    fig.text(1.0,0.0,'Updated '+str(dtime.date())+' ', ha='right',va='bottom', fontsize=8)

def add_signature(fig,url_line):
    by_line = 'Graphics by John Sibert'
    fig.text(0.0,0.020,' '+by_line, ha='left',va='bottom', fontsize=8,alpha=0.25)#,color='red')
    fig.text(1.0,0.020,url_line+' ', ha='right',va='bottom', fontsize=8,alpha=0.25)#,color='red')

def prop_scale(lim,prop):
    s = lim[0] + prop*(lim[1]-lim[0])
    return(s)

def vline(ax, x, label=None, ylim=None, pos='center'):
    if ylim is None:
        ylim = ax.get_ylim()
    ax.plot((x,x), ylim, linewidth=1, linestyle=':')
    c = ax.get_lines()[-1].get_color()
    ax.text(x,ylim[1], label, ha=pos,va='bottom', linespacing=1.8,
            fontsize=8, color=c)


