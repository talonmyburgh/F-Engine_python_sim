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
from time import time



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

s = np.arange(2.5,12.5,1)
x = np.arange(point)
outputpowfloat = np.zeros(20)
outputpowfix = outputpowfloat.copy()
inputpow = np.zeros(20)
st=1
shiftreg = [0,0,0,0,0,0,0,0,0,0]
#shiftreg = [1,1,1,1,1,1,1,1,1,1]
#shiftreg = [0,1,0,1,0,1,0,1,0,1]
#shiftreg = [0,0,0,0,0,1,1,1,1,1]
#shiftreg = [1,1,1,1,1,0,0,0,0,0]

pfb_floating_single = FloatPFB(point,taps)
pfb_fixed_single = FixPFB(point,taps,bits,fraction,shiftreg = shiftreg,method=method)

while st < 11:
    print(st)
    cs1 = np.cos(983.04*np.pi*(n/N),dtype = np.float64)/s[st-1]
    cs2 = np.cos(1966.08*np.pi*(n/N),dtype = np.float64)/s[st-1]
    twotone = cs1+cs2
    fsig1.from_complex(twotone)
    inputpow[st-1] = np.sum(np.abs(twotone)**2)
    pfb_floating_single.run(twotone)
    pfb_fixed_single.run(fsig1)
    outputpowfloat[st-1] = np.sum(np.abs(pfb_floating_single.X_k[:,-1])**2)
    outputpowfix[st-1] = np.sum(np.abs(pfb_fixed_single.X_k.to_complex()[:,-1])**2)
    st+=1

fig = plt.figure()
plt.plot(inputpow, outputpowfloat,'k')
plt.savefig('../snapsf_engine/PFB_twotone/shiftzerod/twotone_power_flt_10.png')
plt.title('Floating point power spectrum: shiftzerod')
plt.xlabel('input power - $\Sigma | input |^{2}$')
plt.ylabel('output power - $\Sigma | output |^{2}$')
plt.show()
fig = plt.figure()
plt.plot(inputpow, outputpowfix,'k')
plt.title('Fixed point power spectrum: shiftzerod')
plt.xlabel('input power - $\Sigma | input |^{2}$')
plt.ylabel('output power - $\Sigma | output |^{2}$')
plt.savefig('../snapsf_engine/PFB_twotone/shiftzerod/twotone_power_fxd_10.png')
plt.show()
#error = np.abs(outputpowfloat-outputpowfix)
    
#pfb_fixed_single.run(fsig1)
#plt.plot(np.abs(pfb_fixed_single.X_k.to_complex()))
#plt.show()
    
#fig, ax = plt.plot(2, 1, sharex=False, sharey=False)
#ax[0].hist(np.abs(pfb_floating_single.X_k[:,-1]),255,color='k')
#ax[0].set_title('floating-point, stage: '+str(st))
#ax[1].hist(np.abs(pfb_fixed_single.X_k.to_complex()[:,-1]),255,color='k')
#ax[1].set_title('fixed-point, stage: '+str(st))
#fig.savefig('../snapsf_engine/PFB_twotone/shiftcanada/unpowered/hist/st'+str(st)+'_twotone_hist.png')
#fig.clear()


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