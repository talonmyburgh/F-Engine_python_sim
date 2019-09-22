#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:15:39 2019

@author: talon
"""

import numpy as np

from fixpoint import cfixpoint,fixpoint
from pfb_fixed import iterffft_natural_DIT as fixFFT
from pfb_fixed import make_fix_twiddle, bitrevfixarray
from pfb_floating_numba import iterfft_natural_in_DIT as floatFFT
from pfb_floating_numba import bitrevarray, make_twiddle


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('%s |%s| %s%% %s' % (prefix, bar, percent, suffix),end="\r")
    # Print New Line on Complete
    if iteration == total: 
        print()

N=2**13
accumval = 500
dmpnum = 0

fixinputev = cfixpoint(17,17,method = "ROUND") #18bit using even rounding
fixtwidsev = make_fix_twiddle(N,17,16,method="ROUND") #18 bit twiddle factor that uses even rounding
fixtwidsev = bitrevfixarray(fixtwidsev,fixtwidsev.data.size)

fixinputinf = cfixpoint(17,17,method = "ROUND_INFTY") #18bit using infty rounding
fixtwidsinf = make_fix_twiddle(N,17,16,method="ROUND_INFTY") #18 bit twiddle factor that uses infty rounding
fixtwidsinf = bitrevfixarray(fixtwidsinf,fixtwidsinf.data.size)

floattwids = make_twiddle(N)
floattwids = bitrevarray(floattwids,floattwids.size)

output_vectorev = fixpoint(31,31,method="ROUND")
output_vectorinf = fixpoint(31,31,method="ROUND")

shiftreg = [1,1,1,1,1,1,0,1,1,1,1,1,1]

i=0
arnge = np.arange(N)

while(True):
    if (i==0):
        printProgressBar(i, accumval, prefix = 'Progress through integration '+str(dmpnum)+':', suffix = 'Complete', length = 50)
        output_vectorev.from_float(np.zeros(2**12,dtype=np.float64))
        output_vectorinf.from_float(np.zeros(2**12,dtype=np.float64))
        output_float = np.zeros(2**12,dtype=np.float64)
    tn = np.sin(280*np.pi*arnge/N,dtype=np.float64)/20
    nse = (np.random.normal(size=N,scale = 0.3)/800).astype(np.float64)
    input_vector = tn+nse
    fixinputev.from_complex(input_vector)
    fixinputinf.from_complex(input_vector)
    
    fftoutev = fixFFT(fixinputev,fixtwidsev,shiftreg.copy(),17,17,17,staged=False)[:2**12]
    outev = fftoutev.power()
    outev.data = outev.data*2**10
    outev.bits=31
    outev.fraction=31
    outev.normalise()
    
    output_vectorev = output_vectorev+outev
    output_vectorev.bits=31
    output_vectorev.fraction=31
    output_vectorev.normalise()
    
    fftoutinf = fixFFT(fixinputinf,fixtwidsinf,shiftreg.copy(),17,17,17,staged=False)[:2**12]
    outinf = fftoutinf.power()
    outinf.data = outinf.data*2**10
    outinf.bits=31
    outinf.fraction=31
    outinf.normalise()
    
    output_vectorinf = output_vectorinf+outinf
    output_vectorinf.bits=31
    output_vectorinf.fraction=31
    output_vectorinf.normalise()
    
    outflt = floatFFT(input_vector,floattwids,staged=False)[:2**12]
    output_float += (outflt*np.conj(outflt)).astype(np.float64)
    
    if (i==accumval):
        np.save("output_vector_evenrounding"+str(dmpnum),output_vectorev.data)
        np.save("output_vector_infiniterounding"+str(dmpnum),output_vectorinf.data)
        np.save("output_vector_float"+str(dmpnum),output_float)
        dmpnum+=1
        i=0
        print("Integration "+str(dmpnum)+" complete!")
    else:
        i+=1
        printProgressBar(i, accumval, prefix = 'Progress through integration '+str(dmpnum)+':', suffix = 'Complete', length = 50)
   
