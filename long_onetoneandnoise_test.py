#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:41:00 2019

@author: talon
"""

import multiprocessing
from pfb_fixed import iterffft_natural_DIT, bitrevfixarray, make_fix_twiddle
import numpy as np
from fixpoint import cfixpoint
import time

N = 2**15 #32k
iters = 1 #100k multiplied by the 10 from running individual processes.
multiple = 92.1 #number of waves per period - purposefully commensurate over 10 x 32k
#we will generate 10 x 32k signals for multiprocessing before abs and summing and saving
resultarray = np.zeros((N,10),dtype = np.float64)

def FFT (ID,DATA,twid,shiftreg,bits,fraction,coeffbits):
    resultarray[:,ID] = np.abs(iterffft_natural_DIT(DATA,twid,shiftreg,bits,fraction,coeffbits).to_complex())


#Generate 10 input signals
sig1 = cfixpoint(22,22) #22 bit fixpoint number that will use even-rounding
sig2 = cfixpoint(22,22)
sig3 = cfixpoint(22,22)
sig4 = cfixpoint(22,22)
sig5 = cfixpoint(22,22)
sig6 = cfixpoint(22,22)
sig7 = cfixpoint(22,22)
sig8 = cfixpoint(22,22)
sig9 = cfixpoint(22,22)
sig10 = cfixpoint(22,22)

tn1 =  ((np.sin(multiple*(2*np.pi*np.arange(N))/N)).astype(np.float64))/20
tn2  = ((np.sin(multiple*(2*np.pi*np.arange(N,2*N))/N)).astype(np.float64))/20
tn3 = ((np.sin(multiple*(2*np.pi*np.arange(2*N,3*N))/N)).astype(np.float64))/20
tn4 = ((np.sin(multiple*(2*np.pi*np.arange(3*N,4*N))/N)).astype(np.float64))/20
tn5 = ((np.sin(multiple*(2*np.pi*np.arange(4*N,5*N))/N)).astype(np.float64))/20
tn6 = ((np.sin(multiple*(2*np.pi*np.arange(5*N,6*N))/N)).astype(np.float64))/20
tn7 = ((np.sin(multiple*(2*np.pi*np.arange(6*N,7*N))/N)).astype(np.float64))/20
tn8 = ((np.sin(multiple*(2*np.pi*np.arange(7*N,8*N))/N)).astype(np.float64))/20
tn9 = ((np.sin(multiple*(2*np.pi*np.arange(8*N,9*N))/N)).astype(np.float64))/20
tn10 = ((np.sin(multiple*(2*np.pi*np.arange(9*N,10*N))/N)).astype(np.float64))/20


shiftreg = [1,1,1,1,0,1,1,1,1,1,0,1,1,1,1]
fixtwids = make_fix_twiddle(2**15,18,17) #18 bit twiddle factor that uses even rounding
fixtwids = bitrevfixarray(fixtwids,fixtwids.data.size)

for i in range(iters):
    #keeps track of the processes
    processes = []
    
    #This is done so that we re-generate new noise vectors every time
    sig1.from_complex(tn1 + ((np.random.normal(size=N)).astype(np.float64)/5000))
    sig2.from_complex(tn2 + ((np.random.normal(size=N)).astype(np.float64)/5000))
    sig3.from_complex(tn3 + ((np.random.normal(size=N)).astype(np.float64)/5000))
    sig4.from_complex(tn4 + ((np.random.normal(size=N)).astype(np.float64)/5000))
    sig5.from_complex(tn5 + ((np.random.normal(size=N)).astype(np.float64)/5000))
    sig6.from_complex(tn6 + ((np.random.normal(size=N)).astype(np.float64)/5000))
    sig7.from_complex(tn7 + ((np.random.normal(size=N)).astype(np.float64)/5000))
    sig8.from_complex(tn8 + ((np.random.normal(size=N)).astype(np.float64)/5000))
    sig9.from_complex(tn9 + ((np.random.normal(size=N)).astype(np.float64)/5000))
    sig10.from_complex(tn10 + ((np.random.normal(size=N)).astype(np.float64)/5000))
    
    sigarray = [sig1,sig2,sig3,sig4,sig5,sig6,sig7,sig8,sig9,sig10]
    
    for j in range(0,10):
        p = multiprocessing.Process(target=FFT, 
                                    args=(j,sigarray[j],fixtwids, shiftreg.copy(), 22,22,22))
        processes.append(p)
        p.start()
        print(j)
        
    for process in processes:
        print(process)
        process.join()
        
        
        
        
        
        
        
        
        
        
        
        
        
        