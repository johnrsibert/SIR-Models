#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:22:41 2020
result = pyreadr.read_r('test_data/basic/two.RData', use_objects=["df1"])
@author: jsibert
"""
import os
import pyreadr
import pandas
import numpy as np
import js_covid as cv

#os.chdir('/home/other/pyreadr')
#result = pyreadr.read_r('test_data/basic/two.RData')
#print(result.keys())
#print(result)
#print('df1:',result['df1'])
#os.chdir('/home/jsibert/Projects/SIR-Models/fits/')

print('----------'+os.getcwd())
#fit = pyreadr.read_r('/home/jsibert/Projects/SIR-Models/fits/Los_AngelesCA.RData')
fit = pyreadr.read_r('/home/jsibert/Projects/SIR-Models/fits/rrSIR/Los_AngelesCA.RData')
#         saving fit: /home/jsibert/Projects/SIR-Models/fits/rrSIR/Los_AngelesCA.RData

#         saving fit: /home/jsibert/Projects/SIR-Models/fits/rrSIR/KingWA.RData
print('keys:',fit.keys())

diag = fit['diag']
print('diag:')
print(diag.columns)
print(diag)
#mbeta = diag['logbeta'].quantile(q=0.5)
#print('median logbeta:',mbeta,np.exp(mbeta))
#mgamma = diag['loggamma'].quantile(q=0.5)
#print('median loggamma:',mgamma,np.exp(mgamma))
mmu = diag['logmu'].quantile(q=0.5)
print('median logmu:',mmu,np.exp(mmu))

md = fit['meta']
md.set_index('names',inplace=True)
print('meta:')
print(md)
#print(md.iloc[0])
#print(md.data)
#rs = md['names'].isin(['Date0'])
#print(rs)
#Date0 = md.data[rs]
#print('Date0:',Date0)

like = fit['like_comp']
print('like:')
print(like)

ests = fit['ests']
ests.set_index('names',inplace=True)
print('ests:')
print(ests)

#print('exp(ests)[est]:')cls
#print(ests['map'])
#print(str(ests['map']))
#print(ests.loc['loggamma']['map'])
#print('   C:',np.exp(ests.loc['logsigma_logC']['est']))
#print('   D:',np.exp(ests.loc['logsigma_logD']['est']))
#print('beta:',np.exp(ests.loc['logsigma_logbeta']['est']))
#print('  mu:',np.exp(ests.loc['logsigma_logmu']['est']))
#print(' map:',       ests.loc['loggamma']['map'])
#print(       ests.loc['loggamma']['map'][0])

#SE = pyreadr.read_r('/home/jsibert/Projects/SIR-Models/fits/stderror.RData')
#print(SE)



#rs = est['names'].isin(['logmu'])
#print(rs)
#if (rs):
#    logmu = est.obs[rs]
#    print('logmu',logmu)
# mx = x.quantile(q=0.5)

