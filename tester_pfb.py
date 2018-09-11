# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:23:55 2018

@author: User
"""

import numpy as np
from pfb_fixed import FixPFB
from pfb_floating import FloatPFB
from fixpoint import cfixpoint
from collections import deque


N = 2048
n = np.arange(N)
bits =18
fraction = 18
method = "round"
taps = 8
point = 128

####SIGNALS######
sig1 = np.cos(1024//6*np.pi*n/N)/2.5
sig2 = np.zeros(N)/2.5
sig2[10:20]=1
sig3 = 1j*sig2+sig2

fsig1 = cfixpoint(bits,fraction,method = method)
fsig2 = cfixpoint(bits,fraction,method = method)
fsig3 = cfixpoint(bits,fraction,method = method)
fsig1.from_complex(sig1)
fsig2.from_complex(sig2)
fsig3.from_complex(sig3)

####SHIFTREGISTER####
shiftreg = deque([0,0,0,0,0,0,0])

pfb_floating_single = FloatPFB(point,taps)
pfb_fixed_single = FixPFB(point,taps,bits,fraction,shiftreg = shiftreg,method=method)
pfb_floating_single.run(sig2)
pfb_fixed_single.run(fsig2)
pfb_floating_single.show()
pfb_fixed_single.show()

pfb_floating_dual = FloatPFB(point,taps,dual = True)
pfb_fixed_dual = FixPFB(point,taps,bits,fraction,shiftreg=shiftreg,method = method, dual = True)
pfb_floating_dual.run(sig3)
pfb_fixed_dual.run(fsig3)
pfb_floating_dual.show()
pfb_fixed_dual.show()