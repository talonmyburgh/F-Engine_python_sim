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
method = "truncate"
taps = 8
point = 128

####SIGNALS######
sig1 = np.cos(2048//3*np.pi*n/N)/2.5
sig2 = np.zeros(N)/2.5
sig2[80:90]=0.2
sig3 = 1j*sig2+np.zeros(N)
rnd1 = (np.random.random(N)-0.5)/2.5
rnd2 = (np.random.random(N)-0.5)/2.5
rnd = rnd1+1j*rnd2


fsig1 = cfixpoint(bits,fraction,method = method)
fsig2 = cfixpoint(bits,fraction,method = method)
fsig3 = cfixpoint(bits,fraction,method = method)
frnd = cfixpoint(bits,fraction,method = method)
fsig1.from_complex(sig1)
fsig2.from_complex(sig2)
fsig3.from_complex(sig3)
frnd.from_complex(rnd)

####SHIFTREGISTER####
shiftreg = deque([0,0,0,0,0,0,0])
#
#pfb_floating_single = FloatPFB(point,taps)
#pfb_fixed_single = FixPFB(point,taps,bits,fraction,shiftreg = shiftreg,method=method)
#pfb_floating_single.run(rnd)
#pfb_fixed_single.run(frnd)
#pfb_floating_single.show()
#pfb_fixed_single.show()

pfb_floating_dual = FloatPFB(point,taps,dual = True)
pfb_fixed_dual = FixPFB(point,taps,bits,fraction,shiftreg=shiftreg,method = method, dual = True)
pfb_floating_dual.run(sig3)
pfb_fixed_dual.run(fsig3)
pfb_floating_dual.show()
pfb_fixed_dual.show()