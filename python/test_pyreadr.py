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
os.chdir('/home/jsibert/Projects/SIR-Models/fits/constrainID/')

print('----------'+os.getcwd())
fit=pyreadr.read_r('MaricopaAZ.RData')
print('keys',fit.keys())
diag = fit['diag']
print(diag.columns)
print(diag)
mmu = diag['logmu'].quantile(q=0.5)
print('median logmu:',mmu)
mbeta = diag['logbeta'].quantile(q=0.5)
print('median logbeta:',mmu)

md = fit['meta']
print('meta:',md)
#print(md.iloc[0])
#print(md.data)
#rs = md['names'].isin(['Date0'])
#print(rs)
#Date0 = md.data[rs]
#print('Date0:',Date0)

like = fit['like_comp']
print(like)

ests = fit['ests']
print('ests:',ests)
print(np.exp(ests['est']))

#rs = est['names'].isin(['logmu'])
#print(rs)
#if (rs):
#    logmu = est.obs[rs]
#    print('logmu',logmu)
# mx = x.quantile(q=0.5)

