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
import matplotlib.pyplot as plt


N = 12288 #12k data points
n = np.arange(N)
bits =18
fraction = 18
method = "truncate"
taps = 8
point = 1024 #1k

####SIGNALS######
#sig1 = np.cos(2048//3*np.pi*n/N)/2.5
#sig2 = np.zeros(N)/2.5
#sig2[80:100]=0.2
#sig3 = 1j*sig2+sig1
#realmeerkat = np.load('noplanes.npy',mmap_mode = 'r')
#sampmeerkat = realmeerkat[4096:12288] #8k
#sampmeerkat = sampmeerkat/100
#rnd1 = (np.random.random(N)-0.5)/2.5
#rnd2 = (np.random.random(N)-0.5)/2.5
#rnd = rnd1+1j*rnd2
cs1 = np.cos(983.04*np.pi*(n/N),dtype = np.float64)/2.5
cs2 = np.cos(1966.08*np.pi*(n/N),dtype = np.float64)/2.5
twotone = cs1+cs2


fsig1 = cfixpoint(bits,fraction,method = method)
#fsig2 = cfixpoint(bits,fraction,method = method)
#fsig3 = cfixpoint(bits,fraction,method = method)
#frnd = cfixpoint(bits,fraction,method = method)
fsig1.from_complex(twotone)
#fsig2.from_complex(sig2)
#fsig3.from_complex(sig3)
#frnd.from_complex(rnd)
#fsampmeerkat = cfixpoint(bits,fraction,method = method)
#fsampmeerkat.from_complex(sampmeerkat)

####SHIFTREGISTER####
st=1
while st <= 10:
    print(st)
#    shiftreg = deque([0,0,0,0,0,0,0,0,0,0]) #test overflow
    shiftreg = deque([1,1,1,1,1,1,1,1,1,1]) #test underflow
    
    pfb_floating_single = FloatPFB(point,taps,staged = st)
    pfb_fixed_single = FixPFB(point,taps,bits,fraction,shiftreg = shiftreg,method=method,staged = st)
    pfb_floating_single.run(twotone)
    pfb_fixed_single.run(fsig1)
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=False)
    ax[0].plot(pfb_floating_single.X_k[:,-1],'k')
    ax[0].set_title('floating-point, stage: '+str(st))
    ax[1].plot(pfb_fixed_single.X_k.to_float()[:,-1],'k')
    ax[1].set_title('fixed-point, stage: '+str(st))
    fig.savefig('../snapsf_engine/PFB_twotone/shiftoned/st'+str(st)+'_twotone.png')
    fig.clear()
    st+=1

#
#b=plt.plot(np.average(pfb_fixed_single.X_k.data,axis = 1))
#plt.ylim((5500,6500))
#plt.show()
#pfb_floating_dual = FloatPFB(point,taps,dual = True)
#pfb_fixed_dual = FixPFB(point,taps,bits,fraction,shiftreg=shiftreg,method = method, dual = True)
#pfb_floating_dual.run(rnd1)
#pfb_fixed_dual.run(rnd)
#pfb_floating_dual.show()
#pfb_fixed_dual.show()