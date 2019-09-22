#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:41:01 2019

@author: talon
"""
import numpy as np

def coeff_gen (N, taps, w='hann', fwidth=1):
    WinDic = {                                                                 #dictionary of various filter types
    'hann' : np.hanning,
    'hamming' : np.hamming,
    'bartlett': np.bartlett,
    'blackman': np.blackman,
    }
    alltaps = N*taps
    windowval=WinDic[w](alltaps)                                               
    totalcoeffs = (windowval*np.sinc(fwidth*(np.arange(alltaps)/(N) - taps/2))).reshape((taps,N)).T
    scalefac = np.ceil(np.log2(np.max(np.sum(np.abs(totalcoeffs),axis=1))))
    return totalcoeffs ,int(scalefac)
