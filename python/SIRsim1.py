#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple SIR model siumation with mortality in the Infected compartment
beta as autocorrelated lognormal random walk

Created on Tue Apr 14 10:03:21 2020

@author: jsibert
"""

import pandas as pd
#from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
#import os
plt.style.use('file:///home/jsibert/.config/matplotlib/john.mplstyle')

# these params are OK
#def SIRsim(mu = 0.001, beta = 0.5, gamma = 0.075, N0 = 1e5, sigma_eye = 0.3,

def SIRsim(mu = 0.001, beta = 0.2, gamma = 0.075, N0 = 2e6, sigma_eye = 0.01,
           ntime = 75):
    """
    Using Wikipedia notation
    """
    time = np.arange(0,ntime,dtype=float)
    S = np.array([0]*ntime,dtype=float)
    Eye = np.array([0]*ntime,dtype=float)
    R = np.array([0]*ntime,dtype=float)
    D = np.array([0]*ntime,dtype=float)
    N = np.array([0]*ntime,dtype=float)
    beta_ts = np.array([0]*ntime,dtype=float)
    eye_error = np.random.lognormal(0.0,sigma_eye,ntime)

    N[0] = N0
    S[0] = N0-1.0
    Eye[0] = 1
    beta_ts[0] = beta

    for t in range(1,len(time)):
        beta_ts[t] = beta_ts[t-1]*eye_error[t]

        N[t] = S[t-1] + Eye[t-1] + R[t-1]
        S[t] = S[t-1] - beta_ts[t]*Eye[t-1]*S[t-1]/N[t-1]
        Eye[t] = Eye[t-1] + (beta_ts[t]*Eye[t-1]*S[t-1]/N[t-1] - gamma*Eye[t-1] - mu*Eye[t-1])
        
        R[t] = R[t-1] + gamma*Eye[t-1]
        D[t] = mu*Eye[t-1]



    state = pd.DataFrame({'N':N, 'D':D, 'R':R, 'Eye':Eye, 'S':S, 'time':time,'beta':beta_ts},
                         dtype=float)
    print(state.head(10))
    print(state.tail(10))
  
    ylim = 1.2*N0
    fig, ax = plt.subplots(3,3,figsize=(18.0,9.0))
    ax[0][0].set_ylabel('Susceptible')
    ax[0][0].plot(time,S)
    ax[0][0].set_ylim(0,ylim)

    ax[0][1].set_ylabel('Infected')
    ax[0][1].plot(time,Eye)

    ax[0][2].set_ylabel('Recovered')
    ax[0][2].plot(time,R)
    ax[0][2].set_ylim(0,ylim)

    ax[1][0].set_ylabel('Population')
    ax[1][0].plot(time,N)
    ax[1][0].set_ylim(0,ylim)

    ax[1][1].set_ylabel('Deaths')
    ax[1][1].plot(time,D)

    ax[1][2].set_ylabel('Pop + Deaths')
    ax[1][2].plot(time,N+D)
    ax[1][2].set_ylim(0,ylim)

    ax[2][0].set_ylabel('beta error')
    ax[2][0].plot(time,eye_error,lw=2)
    ax[2][0].set_ylim(0,max(eye_error))

    ax[2][1].set_ylabel('beta')
    ax[2][1].plot(time,beta_ts,lw=3)
    ax[2][1].set_ylim(0,max(beta_ts))


    plt.show()#block=False)

if __name__ == '__main__':
    SIRsim()

#else:
#    print('type something')

