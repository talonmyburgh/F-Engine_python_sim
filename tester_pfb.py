# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:23:55 2018

@author: User
"""

import numpy as np
from pfb_fixed import FixPFB
from pfb_floating import FloatPFB
from fixpoint import cfixpoint,fixpoint
from collections import deque


N = 1024
n = np.arange(N)
bits =18
fraction = 18
method = "truncate"

####SIGNALS######
sig1 = np.cos(1024//6*np.pi*n/N)/2.5
sig2 = np.zeros(N)/2.5
sig2[10:20]=1
sig3 = sig1+1j*sig2

fsig1 = cfixpoint(bits,fraction,method = method)
fsig2 = cfixpoint(bits,fraction,method = method)
fsig3 = cfixpoint(bits,fraction,method = method)
fsig1.from_complex(sig1)
fsig2.from_complex(sig2)
fsig3.from_complex(sig3)

####SHIFTREGISTER####
shiftreg = deque([1,1,1,1,1,1])

pfb_floating_single = FloatPFB(64,8)
pfb_fixed_single = FixPFB(64,8,bits,fraction,shiftreg = shiftreg,method=method)
pfb_floating_single.run(sig1)
pfb_fixed_single.run(sig1)
pfb_floating_single.show()
pfb_fixed_single.show()