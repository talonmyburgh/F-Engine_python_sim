#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:16:52 2019

@author: talon
"""

from timeit import timeit as t
import numba as nb
import numpy as np

@nb.jit(nopython = True)
def roundinfty(array):
    a = array.copy()
    f=np.modf(a)[0]                                                        #get decimal values from data
    if (a.ndim == 1):                                                      #for 1D array
        for i in range(len(array)):
            if((f[i]<0.0 and f[i] <=-0.5) or (f[i]>=0.0 and f[i]<0.5)):
                a[i]=np.floor(a[i])
            else:
                a[i]=np.ceil(a[i])
    elif(a.ndim==2):                                                       #for 2D array
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if((f[i,j]<0.0 and f[i,j] <=-0.5) or (f[i,j]>=0.0
                   and f[i,j]<0.5)):
                    a[i,j]=np.floor(a[i,j])
                else:
                    a[i,j]=np.ceil(a[i,j])
    elif(a.ndim==3):                                                       #for 3D array
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                for k in range(array.shape[2]):
                    if((f[i,j,k]<0.0 and f[i,j,k] <=-0.5) or 
                       (f[i,j,k]>=0.0 and f[i,j,k]<0.5)):
                        a[i,j,k]=np.floor(a[i,j,k])
                    else:
                        a[i,j,k]=np.ceil(a[i,j,k])
    return a

upset = '''from pfb_floating_numba import FloatPFB
import numpy as np
data = np.ones(8*2**13)
pfbflt = FloatPFB(2**13,8, chan_acc = True)'''

code = '''pfbflt.run(data)'''

print(t(stmt=code,setup=upset,number=1000))