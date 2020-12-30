import config as cv

from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os

def make_date_axis(ax, first_prev_date = None):
    if (first_prev_date is None):
        firstDate = mdates.date2num(FirstNYTDate)
    else:
        firstDate = first_prev_date
        
    lastDate  = mdates.date2num(cv.EndOfTime)
#   print('firstDate,lastDate:',firstDate,lastDate)
    ax.set_xlim([firstDate,lastDate])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator())

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
#   print('len(ax.get_lines())',len(ax.get_lines()))
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


def add_order_date(ax):
#   Newsome's shelter in place order
    orderDate = mdates.date2num(cv.CAOrderDate)
    ax.plot((orderDate,orderDate),
            (ax.get_ylim()[0], ax.get_ylim()[1]),
    #       (0, ax.get_ylim()[1]),
            color='0.5', linewidth=3,alpha=0.5)

def add_data_source(fig,source=None):
    if source is None:
        source = 'New York Times, https://github.com/nytimes/covid-19-data.git.'
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



