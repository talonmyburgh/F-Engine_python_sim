#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:05:49 2019

@author: talon
"""
import numpy as np
"""This method enables wrapping for specified bounds"""
def wrap(array, min_val, max_val):
    a=array.copy()
    if(a.ndim==1):
        b = np.zeros(len(a))
        for i in range(len(a)):
            b[i]=((a[i] - min_val) % ((max_val) - min_val)) + min_val
        return b
    elif(a.ndim==2):
        b = np.zeros((a.shape[0],a.shape[1]))
        for i in range((a.shape[0])):
            for j in range(a.shape[1]):
                b[i,j]=((a[i,j] - min_val) % ((max_val) - min_val)) + max_val
    elif(a.ndim==3):
        b = np.zeros((a.shape[0],a.shape[1],a.shape[2]))
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):
                    b[i,j,k]=((a[i,j,k] - min_val) % ((max_val) - min_val))
                    + min_val
    return b
                    

