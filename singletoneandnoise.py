#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 20:03:43 2019

@author: talon
"""

import numpy as np

iters = 1e2
N = int(2**16*iters)
noise = ((np.random.normal(size=N,scale = 0.3)).astype(np.float64)/15000)
tone = ((np.cos(7000*2*np.pi*np.arange(N)/N).astype(np.float64)+0.5)/20)
input_vector = noise+tone

from fixpoint import fixpoint, cfixpoint
from pfb_fixed import iterffft_natural_DIT as fixFFT
from pfb_fixed import make_fix_twiddle, bitrevfixarray, bit_rev



#RUN 22 bit twiddle
fixinput = cfixpoint(22,22) #22 bit fixpoint number that will use even-rounding
fixtwids18 = make_fix_twiddle(2**16,18,17) #18 bit twiddle factor that uses even rounding
fixtwids18 = bitrevfixarray(fixtwids18,fixtwids18.data.size)

fixtwids22 = make_fix_twiddle(2**16,22,21) #22 bit twiddle factor that uses even rounding
fixtwids22 = bitrevfixarray(fixtwids22,fixtwids22.data.size)

shiftreg = [1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1]
output_vector18 = np.zeros(2**15,dtype=np.float64)
output_vector22 = np.zeros(2**15,dtype=np.float64)



for i in range(iters):
    print(i)
    slc = slice(int(i*2**16),int(i*2**16+2**16),1)
    fixinput.from_complex(input_vector[slc])
    out18= np.abs(fixFFT(fixinput,fixtwids18,shiftreg.copy(),22,22,18,staged=False).to_complex())[:2**15]
    output_vector18 += out18
    out22= np.abs(fixFFT(fixinput,fixtwids22,shiftreg.copy(),22,22,22,staged=False).to_complex())[:2**15]
    output_vector22 += out22

np.save("output_vector_18bit_twiddle",output_vector18)

np.save("output_vector_22bit_twiddle",output_vector22)