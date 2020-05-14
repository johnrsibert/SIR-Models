#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:03:18 2020

@author: jsibert

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import BoxStyle
plt.style.use('file:///home/jsibert/.config/matplotlib/john.mplstyle')

#plt.style.use('fivethirtyeight')
#import numpy as np

def plot_exp():
#    print(plt.style.available)
    
    tmax = 10
    x = list(range(0,tmax+1))
    y = x
    legend = 'very long legend name xx'
    ll = len(legend)
    print(legend,ll)
 
    fig = plt.figure(figsize=(6.5,3))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,y)
    
    boxstyle = BoxStyle("Round", pad=0)
    props = {'boxstyle': boxstyle,
         'facecolor': 'white',
         'linestyle': 'solid',
         'linewidth': 1,
         'edgecolor': 'red'}
        
    print(props)
    
    fs = [4,6,8,10,12,14,16,18,24,36]
 #   yp = [1,3,5,7,9,11]
    nt = len(fs)
    tb = []
    for p in range(0,nt):
        print(p,fs[p])
        tleg = legend+', '+str(fs[p])+'pt'
        tb.append(ax.text(1,y[p],tleg,va='center',bbox=props,fontsize=fs[p]))

    plt.show()
    
    transf = ax.transData.inverted()
    width = []
    height =[]
    for p in range(0,nt): 
        bb = tb[p].get_window_extent(renderer = fig.canvas.renderer)
#        print(bb)#.transformed(transf))
        bb_dim = bb#.transformed(transf)
        width.append(np.round((bb_dim.x1-bb_dim.x0)/ll,3))
        height.append(np.round(bb_dim.y1-bb_dim.y0,3))
    
    print('sizes',fs)
    print('widths',width)
    print('heights',height)     
    for p in range(0,nt):
        print(p,fs[p],width[p],height[p])
        
    tleg = "  "+legend
    tll = len(tleg)
    print('test legend length',tll)
    p=4
    dull = tll*width[p]
    print('in diisplay units',dull,' for size',fs[p])
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot(x,y) 

    ax2.text(x[tmax],y[tmax],tleg,va='center',fontsize=fs[p])
    """
    xldat = ax2.get_xlim()
    print('data',xldat)
    xldisp = ax2.transData.transform(xldat)
    print('display',xldisp)
    xldisp = xldisp + [0.0,dull]
    print('display',xldisp)    
    inv = ax2.transData.inverted()
    xldat = inv.transform(xldisp)
    print('data',xldat)
    new_xldat = ax2.set_xlim(xldat)
    print('new data',new_xldat)
    fig2.suptitle('SF Bay Area Covid-19 Prevalence',va='center',y=0.95,
                clip_on=False)
#    fig2.text(0.5,1.0,'SF Bay Area Covid-19 Prevalence',ha='center',va='top')
    fig2.text(0.5,0.05,'Data source: New York Times',ha='center',va='center',
             fontsize=10)
   
    
    
    plt.show()
    fig2.savefig('myplot.png',dpi=300)
    
    pitch = {}
    tpitch = {}
#   for p in range(0,nt):
#        pitch['size'] = fs[p],{'width',width[p]},{'height',height[p]}
#       pitch[fs[p]] = ({'width',width[p]})/ #,('height',height[p])
#        pitch[fs[p]].update({'width':width[p]})
#        pitch[fs[p]]['width'] = width[p] 
#       people[3]['name'] = 'Luna'
    
#   print(tpitch)
#    print(pitch)
#    
#    for p in range(0,nt):
#        tmp = pitch[fs[p]]
#        print(p,fs[p],tmp['width'])
#        print(p,fs[p],pitch[fs[p]]['width'])
    
#    print( textbox.get_bbox_patch())
#    tbw = textbox.get_bbox_patch().get_width() # 54.121092459652573
#    tbh = textbox.get_bbox_patch().get_height() # 54.121092459652573
#    print('tbw,tbh:',tbw,tbh)#, tbw.get_width())
 
#        print('textbox: ',type(textbox),textbox.get_bbox_patch().get_width()) 
# t = ax.text(x, y, text_string, prop_dict, 
#              bbox=dict(facecolor='red', alpha=0.5, boxstyle='square))
# print t._get_bbox_patch() 
    
    
#    tt = ax.transData.transform((5, 0))
#    print('1: ',tt)
#    xl = ax.get_xlim()
#    print('data',xl)
    
#    tt = ax.transData.transform(xl)
#    print('display',tt)
    
##    wtt = tt + (0,tbw)
#    print('wide display',wtt)
    
#    inv = ax.transData.inverted()
#    wxl = inv.transform(wtt)
#    print('wide xl',wxl)  
#    
#    nxl = ax.set_xlim(xl[0],wxl[1])
#    print('new xl',nxl)
    
#    yl = ax.get_ylim()
#    ax.plot((nxl[1],yl[0]), (nxl[1],yl[1]),color='red')
#    print((nxl[1],yl[0]), (nxl[1],yl[1]))
#    ax.plot(t, x)
#    plt.show()
#    ax.set_xlim(xl[0],xl[1]+ll)
#   xl = ax.get_xlim()

#    print(xl)
#    ax.plot(t, np.log10(x), color='tab:orange')
#
#ax.plot(xdata1, ydata1, color='tab:blue')
#ax.plot(xdata2, ydata2, color='tab:orange')
#   fig.suptitle('SF Bay Area Covid-19 Prevalence',va='center',y=1.0,
#                clip_on=False)
#    fig.text(0.5,1.0,'SF Bay Area Covid-19 Prevalence',ha='center',va='top')
#    fig.text(0.5,0.95,'Data source: New York Times',ha='center',va='center',
#             fontsize=10)
   
    """
    
    plt.savefig('test.png',dpi=300)
    
if __name__ == '__main__':
#    print(mpl.get_configdir())
#    print(plt.style.available)
    plot_exp()

else:
    print('type something')

